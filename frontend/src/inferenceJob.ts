import { api, ApiError } from "./api.ts";
import type { Case, Job } from "./api.ts";

export type JobConnection = {
  kind: "retrying" | "missing" | "paused";
  message: string;
} | null;

type Options = {
  jobId: string;
  isCurrent: () => boolean;
  onJob: (job: Job) => void;
  onConnection: (connection: JobConnection) => void;
  onFinished: (job: Job, jobs: Job[], updated?: Case) => void;
  request?: typeof api;
  pollInterval?: number;
  requestTimeout?: number;
};

export const isActiveJob = (job: Job) =>
  ["queued", "running", "pending"].includes(job.status);
export const isCompletedJob = (job: Job) =>
  ["completed", "succeeded", "done"].includes(job.status);

// One monitor owns one timer and one request batch. Completion is not settled
// until its stored result and history have also been read successfully.
export function monitorInferenceJob({
  jobId,
  isCurrent,
  onJob,
  onConnection,
  onFinished,
  request = api,
  pollInterval = 700,
  requestTimeout = 15000,
}: Options) {
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let controller: AbortController | undefined;
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let terminalJob: Job | undefined;
  let failures = 0;
  const current = () => !stopped && isCurrent();
  const stop = () => {
    stopped = true;
    clearTimeout(timer);
    clearTimeout(timeout);
    controller?.abort();
  };
  const schedule = (delay: number) => {
    if (current()) timer = setTimeout(() => void poll(), delay);
  };
  const poll = async () => {
    if (!current()) return;
    controller = new AbortController();
    const signal = controller.signal;
    timeout = setTimeout(() => controller?.abort(), requestTimeout);
    try {
      const job = terminalJob || await request<Job>(`/jobs/${jobId}`, { signal });
      if (!current()) return;
      if (!terminalJob) onJob(job);
      if (isActiveJob(job)) {
        failures = 0;
        onConnection(null);
        schedule(pollInterval);
        return;
      }
      terminalJob = job;
      // Wait for both requests to settle before retrying; a rejected sibling
      // must not leave an older result request running alongside the next one.
      const [caseResult, historyResult] = await Promise.allSettled([
        isCompletedJob(job)
          ? request<Case>(`/cases/${job.case_id}`, { signal })
          : Promise.resolve(undefined),
        request<{ jobs: Job[] }>("/jobs", { signal }),
      ]);
      if (!current()) return;
      if (caseResult.status === "rejected") throw caseResult.reason;
      if (historyResult.status === "rejected") throw historyResult.reason;
      onConnection(null);
      onFinished(job, historyResult.value.jobs, caseResult.value);
    } catch (error) {
      if (!current()) return;
      const status = error instanceof ApiError ? error.status : undefined;
      if (status === 404) {
        onConnection({
          kind: "missing",
          message: "작업 또는 결과를 찾을 수 없어 자동 조회를 멈췄습니다. 서버 재시작 등으로 기록이 사라졌을 수 있습니다. 다시 연결하면 작업 목록과 저장된 마스크를 확인합니다.",
        });
      } else if (status && status < 500 && ![408, 429].includes(status)) {
        onConnection({
          kind: "paused",
          message: `작업 조회를 멈췄습니다 (${status}). 서버 상태를 확인한 뒤 다시 연결하세요.`,
        });
      } else {
        const delay = Math.min(1000 * 2 ** Math.min(failures++, 3), 8000);
        onConnection({
          kind: "retrying",
          message: `${terminalJob ? "결과와 기록을 불러오지 못했습니다." : "작업 상태에 연결하지 못했습니다."} 마지막으로 확인한 상태를 유지하며 ${delay / 1000}초 후 다시 확인합니다.`,
        });
        schedule(delay);
      }
    } finally {
      clearTimeout(timeout);
      controller = undefined;
    }
  };
  schedule(pollInterval);
  return stop;
}
