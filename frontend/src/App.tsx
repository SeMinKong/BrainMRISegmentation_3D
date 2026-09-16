import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { LoaderCircle, PanelLeft, Upload, X } from "lucide-react";
import { api } from "./api";
import type { Case, Job, Model, Segmentation, Stats } from "./api";
import { isActiveJob, monitorInferenceJob } from "./inferenceJob";
import type { JobConnection } from "./inferenceJob";
import { findPatient, groupPatients } from "./patients";
import CaseHeader from "./components/CaseHeader";
import ExplainView from "./components/ExplainView";
import type { Shown, ViewState } from "./components/ExplainView";
import { Glossary, Help } from "./components/Help";
import ImportDialog from "./components/ImportDialog";
import PatientBrowser from "./components/PatientBrowser";
import ResultCard from "./components/ResultCard";

const defaultView = (data?: Case): ViewState => ({
  modality: data?.modalities.includes("t1c") ? "t1c" : data?.modalities[0] ?? "t1c",
  visibleLabels: data?.labels.map((l) => l.id) ?? [1, 2, 3, 4],
  opacity: 0.6,
  overlay: true,
  isolated: false,
  exploded: false,
  brainOpacity: 0.32,
  contrast: 100,
  resetKey: 0,
  focusKey: 0,
});

/** The newest model output on a case; demo fixtures count on the synthetic case only. */
const latestPrediction = (data?: Case): Segmentation | null =>
  [...(data?.segmentations ?? [])].reverse().find((s) => s.kind === "prediction" || (data?.demo && s.kind === "demo")) ?? null;

export default function App() {
  const [cases, setCases] = useState<Case[]>([]);
  const [caseId, setCaseId] = useState("");
  const [models, setModels] = useState<Model[]>([]);
  const [modelId, setModelId] = useState("");
  const [shown, setShown] = useState<Shown>("reference");
  const [view, setView] = useState<ViewState>(defaultView());
  const [referenceStats, setReferenceStats] = useState<Stats | null>(null);
  const [predictionStats, setPredictionStats] = useState<Stats | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const [monitoredJobId, setMonitoredJobId] = useState<string | null>(null);
  const [jobConnection, setJobConnection] = useState<JobConnection>(null);
  const [reconnecting, setReconnecting] = useState(false);
  const [monitorRevision, setMonitorRevision] = useState(0);
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
  const patient = findPatient(patients, caseId);
  const prediction = latestPrediction(data);
  // The model used for this case: the synthetic pipeline on the phantom, otherwise the first connected checkpoint.
  const model = useMemo(() => {
    if (data?.demo) return models.find((m) => m.demo_only) ?? models.find((m) => m.id === modelId);
    return models.find((m) => m.id === modelId && !m.demo_only) ?? models.find((m) => m.available && !m.demo_only) ?? models.find((m) => !m.demo_only);
  }, [models, modelId, data?.demo]);
  const ready =
    !!model?.available && !!data && (model.demo_only ? data.demo :
      !data.demo && data.label_preset === "mu_glioma_post" && ["t1n", "t1c", "t2w", "t2f"].every((m) => data.modalities.includes(m)));
  const active = monitoredJobId !== null;
  useLayoutEffect(() => {
    selectedCaseId.current = caseId;
    monitoredId.current = monitoredJobId;
  }, [caseId, monitoredJobId]);

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
      const running = j.jobs.find(isActiveJob);
      setActiveJob(running || null);
      setMonitoredJobId(running?.id || null);
      setJobConnection(null);
      // Start on a case the model has not trained on, so the first result shown is a fair one.
      const unseen = c.cases.find((x) => x.study?.split === "val");
      setCaseId(running?.case_id || unseen?.id || c.cases[0]?.id || "");
      setModelId(running?.model_id || m.models.find((x) => x.available && !x.demo_only)?.id || m.models.find((x) => !x.demo_only)?.id || m.models[0]?.id || "");
    } catch (e) {
      if (!controller.signal.aborted) setError((e as Error).message);
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);
  useEffect(() => {
    void initialize();
    return () => {
      initializationRequest.current?.abort();
      recoveryRequest.current?.abort();
    };
  }, [initialize]);

  useEffect(() => {
    if (!data) return;
    setShown(latestPrediction(data) ? "prediction" : "reference");
    setReferenceStats(null);
    setPredictionStats(null);
    setView((v) => ({ ...defaultView(data), opacity: v.opacity, contrast: v.contrast, resetKey: v.resetKey + 1 }));
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
          if (job.case_id === caseId) setShown("prediction");
          setNotice("예측이 끝났습니다. 아래에서 판독 마스크와 비교해 보세요.");
        } else if (["failed", "error"].includes(job.status)) {
          setError(job.message || "예측에 실패했습니다.");
        }
      },
    });
  }, [monitoredJobId, caseId, monitorRevision]);

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
    if (narrow()) setSidebarOpen(false);
  };

  const predict = async () => {
    if (!data || !model || !ready || active) return;
    setSubmitting(true);
    setError("");
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
  const imported = (c: Case) => {
    setCases((v) => [...v.filter((x) => x.id !== c.id), c]);
    setCaseId(c.id);
    setImporting(false);
    setNotice(`${c.name} 볼륨을 가져왔습니다.`);
  };
  const connected = models.some((m) => m.available && !m.demo_only);
  const thisCaseJobs = jobs.filter((j) => j.case_id === caseId && isActiveJob(j));

  return (
    <div className={`app-shell ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <header className="app-header">
        <div className="brand-group">
          <button className="neu-icon" aria-label={sidebarOpen ? "환자 목록 접기" : "환자 목록 열기"} aria-expanded={sidebarOpen} onClick={() => setSidebarOpen((v) => !v)}>
            <PanelLeft size={17} />
          </button>
          <span className="brand">내 모델 결과 보기 · 뇌 MRI 종양</span>
        </div>
        <div className="header-side">
          <span className={`model-chip ${connected ? "on" : ""}`} title={connected ? "체크포인트가 연결되어 예측할 수 있습니다" : "MRI_UNET_CHECKPOINT를 .env에 설정하세요"}>
            <span className="dot" /> {connected ? "내 모델 연결됨" : "모델 미연결"} <Help term="model" />
          </span>
          <button className="neu-btn" onClick={() => setImporting(true)} aria-label="MRI 가져오기">
            <Upload size={16} /> <span>MRI 가져오기</span>
          </button>
        </div>
      </header>

      <aside className="sidebar" aria-label="환자 목록">
        <PatientBrowser patients={patients} caseId={caseId} onSelect={selectCase} onImport={() => setImporting(true)} />
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
        {!loading && !data && (
          <div className="empty-state">
            <h2>표시할 검사가 없습니다</h2>
            <p>서버가 실행 중인지 확인한 뒤 MRI를 가져오세요.</p>
            <button className="neu-btn primary" onClick={() => setImporting(true)}>MRI 가져오기</button>
          </div>
        )}
        {data && (
          <>
            <CaseHeader
              data={data}
              patient={patient}
              model={model}
              ready={ready}
              prediction={prediction}
              active={active || thisCaseJobs.length > 0}
              submitting={submitting}
              activeJob={activeJob}
              jobConnection={jobConnection}
              onPredict={predict}
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
            />
            <Glossary />
          </>
        )}
        <footer className="app-footer">
          <span>연구·학습용 로컬 도구입니다. 진단이나 치료 판단에 쓰지 마세요.</span>
          <a href="https://www.cancerimagingarchive.net/collection/mu-glioma-post/" target="_blank" rel="noreferrer">MU-Glioma-Post 데이터 출처</a>
        </footer>
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
