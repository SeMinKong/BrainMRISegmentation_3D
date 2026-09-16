import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { BarChart3, LoaderCircle, Maximize2, Minimize2, PanelLeft, ScanSearch, Upload, X } from "lucide-react";
import { api } from "./api";
import type { BatchResult, Case, Job, Model, Overview as OverviewData, Stats } from "./api";
import { isActiveJob, monitorInferenceJob } from "./inferenceJob";
import type { JobConnection } from "./inferenceJob";
import { filterPatients, findPatient, groupPatients, latestPrediction, visibleCaseIds } from "./patients";
import type { PatientFilter } from "./patients";
import CaseHeader from "./components/CaseHeader";
import ExplainView from "./components/ExplainView";
import type { Shown, ViewState } from "./components/ExplainView";
import { Glossary, Help } from "./components/Help";
import ImportDialog from "./components/ImportDialog";
import Overview from "./components/Overview";
import type { BatchProgress } from "./components/Overview";
import PatientBrowser from "./components/PatientBrowser";
import ResultCard from "./components/ResultCard";

type Screen = "overview" | "case";
const SETTINGS_KEY = "mri-viewer-settings";
const CHECKLIST_KEY = "mri-viewer-checklist-dismissed";
type Remembered = Partial<Pick<ViewState, "modality" | "opacity" | "contrast" | "isolated" | "exploded" | "brainOpacity">> & { shown?: Shown };

const readSettings = (): Remembered => {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") as Remembered;
  } catch {
    return {};
  }
};
const remembered = readSettings();
// Deep links: ?case=<id>&view=diff opens one visit directly (for sharing a finding or taking screenshots).
const link = typeof location !== "undefined" ? new URLSearchParams(location.search) : new URLSearchParams();
const linkedCase = link.get("case") || "";
const linkedView = (["reference", "prediction", "diff", "both"] as Shown[]).find((v) => v === link.get("view"));

const defaultView = (data?: Case): ViewState => ({
  modality: remembered.modality && data?.modalities.includes(remembered.modality) ? remembered.modality
    : data?.modalities.includes("t1c") ? "t1c" : data?.modalities[0] ?? "t1c",
  visibleLabels: data?.labels.map((l) => l.id) ?? [1, 2, 3, 4],
  opacity: remembered.opacity ?? 0.6,
  overlay: (remembered.opacity ?? 0.6) > 0,
  isolated: remembered.isolated ?? false,
  exploded: remembered.exploded ?? false,
  brainOpacity: remembered.brainOpacity ?? 0.32,
  contrast: remembered.contrast ?? 100,
  showMissed: true,
  showExtra: true,
  resetKey: 0,
  focusKey: 0,
});

const defaultFilter = (hasSplits: boolean): PatientFilter => ({ query: "", split: hasSplits ? "val" : "all", predictedOnly: false, withCavity: false, sort: "id" });

export default function App() {
  const [cases, setCases] = useState<Case[]>([]);
  const [caseId, setCaseId] = useState("");
  const [screen, setScreen] = useState<Screen>(linkedCase ? "case" : "overview");
  const [models, setModels] = useState<Model[]>([]);
  const [modelId, setModelId] = useState("");
  const [shown, setShown] = useState<Shown>(linkedView ?? (remembered.shown && remembered.shown !== "both" ? remembered.shown : "reference"));
  const [view, setView] = useState<ViewState>(defaultView());
  const [filter, setFilter] = useState<PatientFilter>(defaultFilter(true));
  const [referenceStats, setReferenceStats] = useState<Stats | null>(null);
  const [predictionStats, setPredictionStats] = useState<Stats | null>(null);
  const [overview, setOverview] = useState<OverviewData | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(false);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const [monitoredJobId, setMonitoredJobId] = useState<string | null>(null);
  const [jobConnection, setJobConnection] = useState<JobConnection>(null);
  const [reconnecting, setReconnecting] = useState(false);
  const [monitorRevision, setMonitorRevision] = useState(0);
  const [failure, setFailure] = useState<Job | null>(null);
  const [highlight, setHighlight] = useState(false);
  const [batch, setBatch] = useState<(BatchProgress & { ids: Set<string> }) | null>(null);
  const [presenting, setPresenting] = useState(false);
  const [checklist, setChecklist] = useState(() => {
    try { return localStorage.getItem(CHECKLIST_KEY) !== "1"; } catch { return true; }
  });
  const selectedCaseId = useRef(caseId);
  const monitoredId = useRef(monitoredJobId);
  const initializationRequest = useRef<AbortController | null>(null);
  const recoveryRequest = useRef<AbortController | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [importing, setImporting] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState("");
  const narrow = () => typeof window !== "undefined" && window.matchMedia("(max-width: 900px)").matches;
  const [sidebarOpen, setSidebarOpen] = useState(() => !narrow());

  const data = cases.find((c) => c.id === caseId);
  const patients = useMemo(() => groupPatients(cases), [cases]);
  const visiblePatients = useMemo(() => filterPatients(patients, filter), [patients, filter]);
  const order = useMemo(() => visibleCaseIds(visiblePatients), [visiblePatients]);
  const patient = findPatient(patients, caseId);
  const prediction = latestPrediction(data);
  // The model used for this case: the synthetic pipeline on the phantom, otherwise the first connected checkpoint.
  const model = useMemo(() => {
    if (data?.demo) return models.find((m) => m.demo_only) ?? models.find((m) => m.id === modelId);
    return models.find((m) => m.id === modelId && !m.demo_only) ?? models.find((m) => m.available && !m.demo_only) ?? models.find((m) => !m.demo_only);
  }, [models, modelId, data?.demo]);
  const trainedModel = models.find((m) => m.id === modelId && !m.demo_only) ?? models.find((m) => m.available && !m.demo_only) ?? models.find((m) => !m.demo_only);
  const ready =
    !!model?.available && !!data && (model.demo_only ? data.demo :
      !data.demo && data.label_preset === "mu_glioma_post" && ["t1n", "t1c", "t2w", "t2f"].every((m) => data.modalities.includes(m)));
  const active = monitoredJobId !== null;
  useLayoutEffect(() => {
    selectedCaseId.current = caseId;
    monitoredId.current = monitoredJobId;
  }, [caseId, monitoredJobId]);

  // Remember how the viewer was set up, so the next session opens the same way.
  useEffect(() => {
    try {
      const memo: Remembered = { modality: view.modality, opacity: view.overlay ? view.opacity : 0, contrast: view.contrast, isolated: view.isolated,
        exploded: view.exploded, brainOpacity: view.brainOpacity, shown };
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(memo));
    } catch { /* private mode: nothing to remember */ }
  }, [view.modality, view.opacity, view.overlay, view.contrast, view.isolated, view.exploded, view.brainOpacity, shown]);

  const loadOverview = useCallback(async (id: string, signal?: AbortSignal) => {
    if (!id) return;
    setOverviewLoading(true);
    try {
      const result = await api<OverviewData>(`/overview?model_id=${encodeURIComponent(id)}&split=val`, { signal });
      if (!signal?.aborted) setOverview(result);
    } catch (e) {
      if (!signal?.aborted) setError((e as Error).message);
    } finally {
      if (!signal?.aborted) setOverviewLoading(false);
    }
  }, []);

  const initialize = useCallback(async () => {
    initializationRequest.current?.abort();
    const controller = new AbortController();
    initializationRequest.current = controller;
    const options = { signal: controller.signal };
    setLoading(true);
    setError("");
    try {
      const [c, m, j] = await Promise.all([
        api<{ cases: Case[] }>("/cases", options),
        api<{ models: Model[] }>("/models", options),
        api<{ jobs: Job[] }>("/jobs", options),
      ]);
      if (controller.signal.aborted) return;
      setCases(c.cases);
      setModels(m.models);
      setJobs(j.jobs);
      setFilter(defaultFilter(c.cases.some((x) => x.study?.split)));
      const running = j.jobs.find(isActiveJob);
      setActiveJob(running || null);
      setMonitoredJobId(running?.id || null);
      setJobConnection(null);
      // Start on a case the model has not trained on, so the first result shown is a fair one.
      const unseen = c.cases.find((x) => x.study?.split === "val");
      const linked = c.cases.some((x) => x.id === linkedCase) ? linkedCase : "";
      setCaseId(linked || running?.case_id || unseen?.id || c.cases[0]?.id || "");
      const trained = running?.model_id || m.models.find((x) => x.available && !x.demo_only)?.id || m.models.find((x) => !x.demo_only)?.id || m.models[0]?.id || "";
      setModelId(trained);
      void loadOverview(trained, controller.signal);
      // A batch left running by an earlier session keeps being tracked.
      const pending = j.jobs.filter(isActiveJob);
      if (pending.length > 1) setBatch({ ids: new Set(pending.map((x) => x.id)), total: pending.length, done: 0, failed: 0, running: true });
    } catch (e) {
      if (!controller.signal.aborted) setError((e as Error).message);
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, [loadOverview]);
  useEffect(() => {
    void initialize();
    return () => {
      initializationRequest.current?.abort();
      recoveryRequest.current?.abort();
    };
  }, [initialize]);

  useEffect(() => {
    if (!data) return;
    const latest = latestPrediction(data);
    setShown((s) => (latest ? (s === "reference" ? "prediction" : s) : "reference"));
    setReferenceStats(null);
    setPredictionStats(null);
    setFailure(null);
    setHighlight(false);
    setView((v) => ({ ...defaultView(data), modality: data.modalities.includes(v.modality) ? v.modality : defaultView(data).modality,
      opacity: v.opacity, overlay: v.overlay, contrast: v.contrast, isolated: v.isolated, exploded: v.exploded, brainOpacity: v.brainOpacity, resetKey: v.resetKey + 1 }));
  }, [caseId]); // case transitions, not job refreshes

  const predictionId = prediction?.id ?? "";
  useEffect(() => {
    if (!data || !data.segmentations.length) {
      setReferenceStats(null);
      setPredictionStats(null);
      return;
    }
    const controller = new AbortController();
    const reference = data.segmentations.some((s) => s.id === "reference") ? "reference" : data.segmentations[0].id;
    const requests = [api<Stats>(`/cases/${data.id}/stats?segmentation=${reference}&reference=${reference}`, { signal: controller.signal })];
    if (predictionId) requests.push(api<Stats>(`/cases/${data.id}/stats?segmentation=${encodeURIComponent(predictionId)}&reference=${reference}`, { signal: controller.signal }));
    Promise.all(requests)
      .then(([ref, pred]) => {
        if (controller.signal.aborted) return;
        setReferenceStats(ref);
        setPredictionStats(pred ?? null);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => controller.abort();
  }, [data?.id, predictionId, data?.segmentations.length]);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 6000);
    return () => clearTimeout(timer);
  }, [notice]);
  useEffect(() => {
    if (!highlight) return;
    document.getElementById("result-card")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    const timer = setTimeout(() => setHighlight(false), 2400);
    return () => clearTimeout(timer);
  }, [highlight]);

  useEffect(() => {
    if (!monitoredJobId || jobConnection?.kind === "missing" || jobConnection?.kind === "paused") return;
    return monitorInferenceJob({
      jobId: monitoredJobId,
      isCurrent: () => selectedCaseId.current === caseId && monitoredId.current === monitoredJobId,
      onJob: setActiveJob,
      onConnection: setJobConnection,
      onFinished: (job, allJobs, updated) => {
        setJobs(allJobs);
        setMonitoredJobId(null);
        if (updated) {
          setCases((v) => v.map((c) => (c.id === updated.id ? updated : c)));
          if (job.case_id === caseId) {
            setShown("prediction");
            setHighlight(true);
          }
          setNotice("예측이 끝났습니다. 일치도와 차이 보기를 확인해 보세요.");
          void loadOverview(job.model_id);
        } else if (["failed", "error"].includes(job.status)) {
          setFailure(job);
        }
      },
    });
  }, [monitoredJobId, caseId, monitorRevision]);

  // Batch prediction: the server runs one job at a time; we poll the job list and refresh finished cases.
  useEffect(() => {
    if (!batch?.running) return;
    let cancelled = false;
    const seen = new Set<string>();
    const tick = async () => {
      try {
        const { jobs: all } = await api<{ jobs: Job[] }>("/jobs");
        if (cancelled) return;
        setJobs(all);
        const mine = all.filter((j) => batch.ids.has(j.id));
        const finished = mine.filter((j) => !isActiveJob(j));
        const fresh = finished.filter((j) => !seen.has(j.id) && j.status === "completed");
        fresh.forEach((j) => seen.add(j.id));
        if (fresh.length) {
          const updates = await Promise.all(fresh.map((j) => api<Case>(`/cases/${j.case_id}`).catch(() => null)));
          if (cancelled) return;
          setCases((v) => v.map((c) => updates.find((u) => u?.id === c.id) ?? c));
        }
        const done = finished.length + (batch.total - mine.length); // jobs the server already forgot count as done
        const failed = finished.filter((j) => j.status === "failed").length;
        const running = mine.some(isActiveJob);
        setBatch((b) => (b ? { ...b, done, failed, running } : b));
        if (!running) {
          setNotice(`전체 예측이 끝났습니다. ${done - failed}개 성공${failed ? `, ${failed}개 실패` : ""}.`);
          void loadOverview(modelId);
        }
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    };
    void tick();
    const timer = setInterval(() => void tick(), 1500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [batch?.running, batch?.ids]);

  const reconnectJobs = async () => {
    recoveryRequest.current?.abort();
    const controller = new AbortController();
    recoveryRequest.current = controller;
    setReconnecting(true);
    try {
      const [caseResult, jobResult] = await Promise.all([
        api<{ cases: Case[] }>("/cases", { signal: controller.signal }),
        api<{ jobs: Job[] }>("/jobs", { signal: controller.signal }),
      ]);
      if (controller.signal.aborted) return;
      setCases(caseResult.cases);
      setJobs(jobResult.jobs);
      setCaseId((current) => (caseResult.cases.some((c) => c.id === current) ? current : caseResult.cases[0]?.id || ""));
      const job = jobResult.jobs.find((j) => j.id === monitoredId.current) ||
        jobResult.jobs.find((j) => j.case_id === selectedCaseId.current && isActiveJob(j)) ||
        jobResult.jobs.find(isActiveJob);
      setActiveJob(job || null);
      setMonitoredJobId(job?.id || null);
      setJobConnection(null);
      setMonitorRevision((v) => v + 1);
      if (!job) setNotice("작업 목록과 저장된 마스크를 다시 불러왔습니다. 실행 중인 예측은 없습니다.");
    } catch (e) {
      if (!controller.signal.aborted) setJobConnection({ kind: "paused", message: `다시 연결하지 못했습니다. ${(e as Error).message}` });
    } finally {
      if (!controller.signal.aborted) setReconnecting(false);
    }
  };

  const patchView = (patch: Partial<ViewState>) => setView((v) => ({ ...v, ...patch }));
  const toggleLabel = (id: number) =>
    setView((v) => ({ ...v, visibleLabels: v.visibleLabels.includes(id) ? v.visibleLabels.filter((n) => n !== id) : [...v.visibleLabels, id] }));
  const reset = () => setView((v) => ({ ...defaultView(data), modality: v.modality, resetKey: v.resetKey + 1 }));
  const focusTumor = () => setView((v) => ({ ...v, focusKey: v.focusKey + 1 }));
  const selectCase = (id: string) => {
    setCaseId(id);
    setScreen("case");
    if (narrow()) setSidebarOpen(false);
  };
  const position = order.indexOf(caseId);
  const previous = position > 0 ? () => selectCase(order[position - 1]) : undefined;
  const next = position >= 0 && position < order.length - 1 ? () => selectCase(order[position + 1]) : undefined;

  // Keyboard: ← → step through the list, Esc leaves presentation mode. Ignored while typing or scrubbing a slice.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName) || target.isContentEditable)) return;
      if (event.key === "Escape" && presenting) setPresenting(false);
      if (screen !== "case" || event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.key === "ArrowLeft" && previous) { event.preventDefault(); previous(); }
      if (event.key === "ArrowRight" && next) { event.preventDefault(); next(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [previous, next, presenting, screen]);

  const predict = async () => {
    if (!data || !model || !ready || active) return;
    setSubmitting(true);
    setError("");
    setFailure(null);
    try {
      const job = await api<Job>("/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case_id: caseId, model_id: model.id }),
      });
      setActiveJob(job);
      setMonitoredJobId(job.id);
      setJobConnection(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };
  const batchPredict = async () => {
    if (!trainedModel?.available || batch?.running) return;
    const ids = cases.filter((c) => c.study?.split === "val" && !c.demo && c.segmentations.some((s) => s.id === "reference")).map((c) => c.id);
    if (!ids.length) return;
    setError("");
    try {
      const result = await api<BatchResult>("/jobs/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: trainedModel.id, case_ids: ids, skip_predicted: true }),
      });
      if (!result.queued) {
        setNotice(`새로 예측할 검사가 없습니다. ${result.skipped.length}개는 이미 예측되어 있거나 건너뛰었습니다.`);
        return;
      }
      setBatch({ ids: new Set(result.jobs.map((j) => j.id)), total: result.queued, done: 0, failed: 0, running: true });
      setNotice(`${result.queued}개 검사를 순서대로 예측합니다. 진행 중에도 다른 검사를 볼 수 있습니다.`);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const imported = (c: Case) => {
    setCases((v) => [...v.filter((x) => x.id !== c.id), c]);
    selectCase(c.id);
    setImporting(false);
    setNotice(`${c.name} 볼륨을 가져왔습니다.`);
  };
  const dismissChecklist = () => {
    setChecklist(false);
    try { localStorage.setItem(CHECKLIST_KEY, "1"); } catch { /* ignore */ }
  };
  const connected = models.some((m) => m.available && !m.demo_only);
  const thisCaseJobs = jobs.filter((j) => j.case_id === caseId && isActiveJob(j));
  const caseFailure = failure ?? [...jobs].reverse().find((j) => j.case_id === caseId && j.status === "failed" && !prediction) ?? null;

  return (
    <div className={`app-shell ${sidebarOpen && !presenting ? "" : "sidebar-collapsed"} ${presenting ? "presenting" : ""}`}>
      <header className="app-header">
        <div className="brand-group">
          <button className="neu-icon" aria-label={sidebarOpen ? "환자 목록 접기" : "환자 목록 열기"} aria-expanded={sidebarOpen} onClick={() => setSidebarOpen((v) => !v)}>
            <PanelLeft size={17} />
          </button>
          <span className="brand">내 모델 결과 보기 · 뇌 MRI 종양</span>
          <nav className="segmented small screen-switch" aria-label="화면">
            <button className={screen === "overview" ? "pressed" : ""} aria-pressed={screen === "overview"} onClick={() => setScreen("overview")}>
              <BarChart3 size={15} /> 모델 성능
            </button>
            <button className={screen === "case" ? "pressed" : ""} aria-pressed={screen === "case"} disabled={!data} onClick={() => setScreen("case")}>
              <ScanSearch size={15} /> 검사 보기
            </button>
          </nav>
        </div>
        <div className="header-side">
          {batch?.running && (
            <span className="batch-chip" role="status" title="전체 예측 진행 중">
              <LoaderCircle className="spin" size={14} /> 전체 예측 {batch.done} / {batch.total}
            </span>
          )}
          <span className={`model-chip ${connected ? "on" : ""}`} title={connected ? "체크포인트가 연결되어 예측할 수 있습니다" : "MRI_UNET_CHECKPOINT를 .env에 설정하세요"}>
            <span className="dot" /> {connected ? "내 모델 연결됨" : "모델 미연결"} <Help term="model" />
          </span>
          {screen === "case" && data && (
            <button className="neu-icon" aria-label={presenting ? "발표 모드 끄기" : "발표 모드"} aria-pressed={presenting} title={presenting ? "발표 모드 끄기 (Esc)" : "발표 모드: 목록과 설명을 숨기고 영상만 크게"} onClick={() => setPresenting((v) => !v)}>
              {presenting ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
            </button>
          )}
          <button className="neu-btn" onClick={() => setImporting(true)} aria-label="MRI 가져오기">
            <Upload size={16} /> <span>MRI 가져오기</span>
          </button>
        </div>
      </header>

      <aside className="sidebar" aria-label="환자 목록">
        <PatientBrowser patients={patients} visible={visiblePatients} filter={filter} onFilter={(patch) => setFilter((f) => ({ ...f, ...patch }))}
          caseId={caseId} onSelect={selectCase} onImport={() => setImporting(true)} />
      </aside>

      <main className="main-content">
        {error && (
          <div className="banner error" role="alert">
            <span>{error}</span>
            <div>
              {!data && <button className="neu-btn" onClick={initialize}>다시 연결</button>}
              <button className="neu-icon" aria-label="오류 알림 닫기" onClick={() => setError("")}><X size={15} /></button>
            </div>
          </div>
        )}
        {jobConnection && (
          <div className="banner" role="status">
            <span>{jobConnection.message}</span>
            {jobConnection.kind !== "retrying" && (
              <button className="neu-btn" onClick={reconnectJobs} disabled={reconnecting}>
                {reconnecting ? "다시 연결 중…" : "작업·결과 다시 연결"}
              </button>
            )}
          </div>
        )}
        {loading && (
          <div className="empty-state">
            <LoaderCircle className="spin" size={28} />
            <h2>환자 목록을 불러오는 중</h2>
          </div>
        )}
        {!loading && !cases.length && (
          <div className="empty-state">
            <h2>표시할 검사가 없습니다</h2>
            <p>서버가 실행 중인지 확인한 뒤 MRI를 가져오세요.</p>
            <button className="neu-btn primary" onClick={() => setImporting(true)}>MRI 가져오기</button>
          </div>
        )}
        {!loading && cases.length > 0 && screen === "overview" && (
          <Overview overview={overview} loading={overviewLoading} model={trainedModel} cases={cases} batch={batch} onBatch={batchPredict}
            onOpenCase={selectCase} showChecklist={checklist} onDismissChecklist={dismissChecklist} />
        )}
        {data && screen === "case" && (
          <>
            <CaseHeader
              data={data}
              patient={patient}
              model={model}
              ready={ready}
              prediction={prediction}
              active={active || thisCaseJobs.length > 0}
              submitting={submitting}
              activeJob={activeJob ?? thisCaseJobs[0] ?? null}
              jobConnection={jobConnection}
              failure={caseFailure}
              onPredict={predict}
              onDismissFailure={() => setFailure(null)}
              position={position >= 0 ? { index: position, total: order.length } : undefined}
              onPrevious={previous}
              onNext={next}
            />
            <ExplainView
              data={data}
              patient={patient}
              lead={
                <ResultCard
                  data={data}
                  prediction={prediction}
                  referenceStats={referenceStats}
                  predictionStats={predictionStats}
                  visibleLabels={view.visibleLabels}
                  onToggleLabel={toggleLabel}
                  overview={overview}
                  highlight={highlight}
                  onShowDiff={prediction && data.segmentations.some((s) => s.id === "reference") ? () => setShown("diff") : undefined}
                />
              }
              prediction={prediction}
              shown={shown}
              onShown={setShown}
              stats={referenceStats}
              predictionStats={predictionStats}
              view={view}
              onView={patchView}
              onFocus={focusTumor}
              onReset={reset}
              onSelectCase={selectCase}
              presenting={presenting}
            />
            {!presenting && <Glossary />}
          </>
        )}
        {!presenting && (
          <footer className="app-footer">
            <span>연구·학습용 로컬 도구입니다. 진단이나 치료 판단에 쓰지 마세요.</span>
            <a href="https://www.cancerimagingarchive.net/collection/mu-glioma-post/" target="_blank" rel="noreferrer">MU-Glioma-Post 데이터 출처</a>
          </footer>
        )}
      </main>

      {importing && <ImportDialog onClose={() => setImporting(false)} onImported={imported} />}
      {notice && (
        <div className="toast" role="status">
          {notice}
          <button className="neu-icon" aria-label="알림 닫기" onClick={() => setNotice("")}><X size={14} /></button>
        </div>
      )}
    </div>
  );
}
