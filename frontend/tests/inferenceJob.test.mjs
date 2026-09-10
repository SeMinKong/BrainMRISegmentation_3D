import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import test from "node:test";
import { api, ApiError } from "../src/api.ts";
import { monitorInferenceJob } from "../src/inferenceJob.ts";

const running = { id: "job-a", case_id: "case-a", model_id: "unet3d", status: "running", progress: 50, message: "Inference" };
const completed = { ...running, status: "completed", progress: 100, segmentation_id: "result-a" };
const updated = { id: "case-a", segmentations: [{ id: "result-a" }] };
const history = { jobs: [completed] };
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const advance = async (t, ms) => {
  t.mock.timers.tick(ms);
  await setImmediate();
};
function watch(t, request, options = {}) {
  const events = { jobs: [], connections: [], finished: [] };
  const stop = monitorInferenceJob({
    jobId: running.id,
    isCurrent: () => true,
    onJob: (job) => events.jobs.push(job),
    onConnection: (connection) => events.connections.push(connection),
    onFinished: (...args) => events.finished.push(args),
    request,
    ...options,
  });
  t.after(stop);
  return { ...events, stop };
}
function timers(t) {
  t.mock.timers.enable({ apis: ["setTimeout"] });
}

test("a temporary status connection failure preserves server status and recovers automatically", async (t) => {
  timers(t);
  const calls = [];
  let statusReads = 0;
  const events = watch(t, async (path) => {
    calls.push(path);
    if (path === "/jobs/job-a") {
      statusReads++;
      if (statusReads === 1) return running;
      if (statusReads === 2) throw new TypeError("Failed to fetch");
      return completed;
    }
    return path === "/jobs" ? history : updated;
  });
  await advance(t, 700);
  await advance(t, 700);
  assert.equal(statusReads, 2);
  assert.deepEqual(events.jobs, [running]);
  assert.equal(events.connections.at(-1).kind, "retrying");
  await advance(t, 999);
  assert.equal(statusReads, 2);
  await advance(t, 1);
  assert.deepEqual(events.jobs, [running, completed]);
  assert.deepEqual(events.finished, [[completed, history.jobs, updated]]);
  assert.equal(events.connections.at(-1), null);
  await advance(t, 60000);
  assert.equal(calls.length, 5);
});

test("HTTP 503 on a status read is retried without inventing a failed job", async (t) => {
  timers(t);
  let calls = 0;
  const events = watch(t, async () => {
    if (++calls === 1) throw new ApiError("Unavailable", 503);
    return running;
  });
  await advance(t, 700);
  assert.deepEqual(events.jobs, []);
  assert.equal(events.connections.at(-1).kind, "retrying");
  await advance(t, 1000);
  assert.deepEqual(events.jobs, [running]);
  assert.equal(events.connections.at(-1), null);
});

for (const failedPath of ["/cases/case-a", "/jobs"]) {
  test(`completed jobs retry failed result synchronization at ${failedPath}`, async (t) => {
    timers(t);
    const counts = new Map();
    const events = watch(t, async (path) => {
      const count = (counts.get(path) || 0) + 1;
      counts.set(path, count);
      if (path === "/jobs/job-a") return completed;
      if (path === failedPath && count === 1) throw new ApiError("Unavailable", 503);
      return path === "/jobs" ? history : updated;
    });
    await advance(t, 700);
    assert.deepEqual(events.jobs, [completed]);
    assert.equal(events.finished.length, 0);
    assert.equal(events.connections.at(-1).kind, "retrying");
    await advance(t, 1000);
    assert.equal(counts.get("/jobs/job-a"), 1, "terminal status survives result read failure");
    assert.equal(counts.get("/cases/case-a"), 2);
    assert.equal(counts.get("/jobs"), 2);
    assert.deepEqual(events.finished, [[completed, history.jobs, updated]]);
  });
}

test("a slow status request cannot overlap with later polls", async (t) => {
  timers(t);
  const pending = deferred();
  let calls = 0;
  const events = watch(t, async () => { calls++; return pending.promise; });
  await advance(t, 700);
  await advance(t, 10000);
  assert.equal(calls, 1);
  pending.resolve(running);
  await setImmediate();
  assert.deepEqual(events.jobs, [running]);
  await advance(t, 699);
  assert.equal(calls, 1);
  await advance(t, 1);
  assert.equal(calls, 2);
});

test("result retry waits for the other result request to settle", async (t) => {
  timers(t);
  const oldHistory = deferred();
  const counts = new Map();
  const events = watch(t, async (path) => {
    const count = (counts.get(path) || 0) + 1;
    counts.set(path, count);
    if (path === "/jobs/job-a") return completed;
    if (path === "/cases/case-a" && count === 1) throw new ApiError("Unavailable", 503);
    if (path === "/jobs" && count === 1) return oldHistory.promise;
    return path === "/jobs" ? history : updated;
  });
  await advance(t, 700);
  await advance(t, 5000);
  assert.equal(counts.get("/cases/case-a"), 1);
  assert.equal(counts.get("/jobs"), 1);
  oldHistory.resolve(history);
  await setImmediate();
  await advance(t, 1000);
  assert.equal(events.finished.length, 1);
  assert.equal(counts.get("/cases/case-a"), 2);
});

test("changing the selected case invalidates a late status response before cleanup", async (t) => {
  timers(t);
  const pending = deferred();
  let selectedCase = "case-a";
  let calls = 0;
  const events = watch(t, async () => { calls++; return pending.promise; }, {
    isCurrent: () => selectedCase === "case-a",
  });
  await advance(t, 700);
  selectedCase = "case-b";
  pending.resolve(completed);
  await setImmediate();
  assert.deepEqual(events.jobs, []);
  assert.deepEqual(events.finished, []);
  await advance(t, 60000);
  assert.equal(calls, 1, "stale status cannot begin result synchronization");
});

test("case switching during completion synchronization blocks old results and aborts both reads", async (t) => {
  timers(t);
  const oldCase = deferred(), oldHistory = deferred();
  const signals = [];
  let selectedCase = "case-a";
  const events = watch(t, async (path, { signal }) => {
    if (path === "/jobs/job-a") return completed;
    signals.push(signal);
    return path === "/jobs" ? oldHistory.promise : oldCase.promise;
  }, { isCurrent: () => selectedCase === "case-a" });
  await advance(t, 700);
  assert.deepEqual(events.jobs, [completed]);
  selectedCase = "case-b";
  events.stop();
  assert.equal(signals.length, 2);
  assert.ok(signals.every((signal) => signal.aborted));
  // Even a transport that resolves after abort cannot update the new case.
  oldCase.resolve(updated);
  oldHistory.resolve(history);
  await setImmediate();
  assert.deepEqual(events.finished, []);
  assert.deepEqual(events.connections, []);
});

test("replacing a monitor ignores the old late response and keeps the replacement active", async (t) => {
  timers(t);
  const pending = deferred();
  let oldSignal;
  const old = watch(t, async (_path, { signal }) => { oldSignal = signal; return pending.promise; });
  await advance(t, 700);
  old.stop();
  const replacement = watch(t, async () => running);
  await advance(t, 700);
  pending.resolve(completed);
  await setImmediate();
  assert.equal(oldSignal.aborted, true);
  assert.deepEqual(old.jobs, []);
  assert.deepEqual(old.finished, []);
  assert.deepEqual(replacement.jobs, [running]);
});

test("cleanup abort errors do not become connection alerts or schedule retries", async (t) => {
  timers(t);
  let calls = 0;
  const events = watch(t, async (_path, { signal }) => {
    calls++;
    return new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    });
  });
  await advance(t, 700);
  events.stop();
  await setImmediate();
  await advance(t, 60000);
  assert.equal(calls, 1);
  assert.deepEqual(events.jobs, []);
  assert.deepEqual(events.connections, []);
});

test("a timed-out status request aborts and automatically retries", async (t) => {
  timers(t);
  let calls = 0, firstSignal;
  const events = watch(t, async (_path, { signal }) => {
    if (++calls > 1) return running;
    firstSignal = signal;
    return new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    });
  });
  await advance(t, 700);
  await advance(t, 15000);
  assert.equal(firstSignal.aborted, true);
  assert.equal(events.connections.at(-1).kind, "retrying");
  await advance(t, 1000);
  assert.deepEqual(events.jobs, [running]);
});

for (const [status, kind] of [[404, "missing"], [403, "paused"]]) {
  test(`HTTP ${status} pauses monitoring without a fake failed job or endless retries`, async (t) => {
    timers(t);
    let calls = 0;
    const events = watch(t, async () => { calls++; throw new ApiError("Cannot read job", status); });
    await advance(t, 700);
    assert.equal(events.connections.at(-1).kind, kind);
    assert.deepEqual(events.jobs, []);
    assert.deepEqual(events.finished, []);
    await advance(t, 60000);
    assert.equal(calls, 1);
  });
}

test("a result that disappeared after completion pauses without changing completed status", async (t) => {
  timers(t);
  let calls = 0;
  const events = watch(t, async (path) => {
    calls++;
    if (path === "/jobs/job-a") return completed;
    if (path === "/jobs") return history;
    throw new ApiError("No case", 404);
  });
  await advance(t, 700);
  assert.deepEqual(events.jobs, [completed]);
  assert.equal(events.connections.at(-1).kind, "missing");
  await advance(t, 60000);
  assert.equal(calls, 3);
});

test("actual server-reported failure remains a failed job and synchronizes its history", async (t) => {
  timers(t);
  const failed = { ...running, status: "failed", message: "Inference failed" };
  const events = watch(t, async (path) => path === "/jobs/job-a" ? failed : { jobs: [failed] });
  await advance(t, 700);
  assert.deepEqual(events.jobs, [failed]);
  assert.deepEqual(events.finished, [[failed, [failed], undefined]]);
});

test("API errors retain the HTTP status needed to distinguish missing jobs from transient failures", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response(JSON.stringify({ detail: "Job not found" }), {
    status: 404,
    headers: { "Content-Type": "application/json" },
  }));
  await assert.rejects(api("/jobs/missing"), (error) =>
    error instanceof ApiError && error.status === 404 && error.message === "Job not found");
});
