import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  BookOpen,
  Box,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Cpu,
  Eye,
  EyeOff,
  Files,
  FlaskConical,
  Focus,
  GitCompareArrows,
  Layers3,
  LoaderCircle,
  PanelLeftClose,
  Play,
  Plus,
  RotateCcw,
  Scan,
  Settings2,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import { api, modalityName, number } from "./api";
import type { Case, Job, Model, Stats } from "./api";
import { isActiveJob, isCompletedJob, monitorInferenceJob } from "./inferenceJob";
import type { JobConnection } from "./inferenceJob";
import MeshViewer from "./components/MeshViewer";
import SliceViewer from "./components/SliceViewer";

type Tab = "workspace" | "compare" | "learn";
const tabs = [
  { id: "workspace", name: "워크스페이스", icon: Box },
  { id: "compare", name: "결과 비교", icon: GitCompareArrows },
  { id: "learn", name: "모델 학습", icon: FlaskConical },
] as const;
function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: () => void;
  label: string;
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      aria-label={label}
      className={`switch ${checked ? "on" : ""}`}
      onClick={onChange}
    >
      <span />
    </button>
  );
}
function ImportDialog({
  onClose,
  onImported,
}: {
  onClose: () => void;
  onImported: (data: Case) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [preset, setPreset] = useState("mu_glioma_post");
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  const accept = (items: File[]) => {
    const accepted = items.filter((f) => /\.nii(\.gz)?$/i.test(f.name));
    setFiles(accepted);
    setError(
      accepted.length === items.length
        ? ""
        : ".nii 또는 .nii.gz 파일만 사용할 수 있습니다.",
    );
  };
  const submit = async () => {
    setBusy(true);
    setError("");
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    form.append("label_preset", preset);
    if (name.trim()) form.append("name", name.trim());
    try {
      onImported(
        await api<Case>("/cases/import", { method: "POST", body: form }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <dialog
      ref={dialog}
      className="import-dialog"
      onCancel={(e) => {
        e.preventDefault();
        if (!busy) onClose();
      }}
      onClick={(e) => {
        if (e.target === dialog.current && !busy) onClose();
      }}
    >
      <div className="dialog-header">
        <div className="dialog-symbol">
          <Upload size={23} />
        </div>
        <button
          className="icon-btn"
          onClick={onClose}
          disabled={busy}
          aria-label="가져오기 창 닫기"
        >
          <X size={20} />
        </button>
      </div>
      <h2>MRI 볼륨 가져오기</h2>
      <p>한 사례의 영상과 분할 마스크를 함께 선택하세요.</p>
      <label className="field-label" htmlFor="case-name">
        사례 이름 <span>선택 사항</span>
      </label>
      <input
        id="case-name"
        className="text-input"
        placeholder="예: 환자001_검사01"
        value={name}
        onChange={(e) => setName(e.target.value)}
        disabled={busy}
      />
      <label
        className="drop-zone"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          if (!busy) accept(Array.from(e.dataTransfer.files));
        }}
      >
        <Files size={30} />
        <strong>파일을 놓거나 클릭하여 선택</strong>
        <span>NIfTI · .nii / .nii.gz · 파일당 최대 256 MB</span>
        <input
          type="file"
          accept=".nii,.gz"
          multiple
          disabled={busy}
          onChange={(e) => accept(Array.from(e.target.files || []))}
        />
      </label>
      {files.length > 0 && (
        <div className="file-list">
          {files.map((f) => (
            <div key={f.name}>
              <Check size={14} />
              <span>{f.name}</span>
              <small>{number(f.size / 1024 / 1024, 1)} MB</small>
            </div>
          ))}
        </div>
      )}
      <label className="field-label" htmlFor="label-preset">
        마스크 라벨 체계
      </label>
      <select
        className="text-input"
        id="label-preset"
        value={preset}
        disabled={busy}
        onChange={(e) => setPreset(e.target.value)}
      >
        <option value="mu_glioma_post">
          MU-Glioma-Post · 1 NETC / 2 SNFH / 3 ET / 4 RC
        </option>
        <option value="generic">사용자 데이터 · 라벨 1–4</option>
      </select>
      <div className="import-guidance" style={{ marginTop: 14 }}>
        <CircleHelp size={17} />
        <div>
          이 프로젝트는 파일 이름의 <code>_t1n</code>, <code>_t1c</code>,{" "}
          <code>_t2w</code>, <code>_t2f</code>, <code>_seg</code> 접미사를
          인식합니다. 원본 이름이 다르면 복사본 이름을 맞춰 가져오세요. 영상
          한 개만 가져올 수도 있습니다. 데이터의 라벨 체계를 선택하세요.
          라벨 번호를 자동 변환하지 않습니다.
        </div>
      </div>
      <p className="privacy-note">
        <ShieldCheck size={14} /> 영상은 이 컴퓨터의 서버에만 저장됩니다.
      </p>
      {error && (
        <div role="alert" className="inline-error">
          {error}
        </div>
      )}
      <div className="dialog-footer">
        <button className="button secondary" onClick={onClose} disabled={busy}>
          취소
        </button>
        <button
          className="button primary"
          disabled={!files.length || busy}
          onClick={submit}
        >
          {busy ? (
            <LoaderCircle className="spin" size={17} />
          ) : (
            <Upload size={17} />
          )}{" "}
          {busy ? "볼륨 검증 중…" : "워크스페이스에 추가"}
        </button>
      </div>
    </dialog>
  );
}

export default function App() {
  const [cases, setCases] = useState<Case[]>([]);
  const [caseId, setCaseId] = useState("");
  const [models, setModels] = useState<Model[]>([]);
  const [modelId, setModelId] = useState("");
  const [tab, setTab] = useState<Tab>("workspace");
  const [segmentation, setSegmentation] = useState("reference");
  const [comparison, setComparison] = useState("demo_prediction");
  const [modality, setModality] = useState("t1c");
  const [visibleLabels, setVisibleLabels] = useState<number[]>([1, 2, 3, 4]);
  const [isolated, setIsolated] = useState(false);
  const [exploded, setExploded] = useState(false);
  const [brainOpacity, setBrainOpacity] = useState(0.13);
  const [overlay, setOverlay] = useState(true);
  const [opacity, setOpacity] = useState(0.6);
  const [contrast, setContrast] = useState(100);
  const [resetKey, setResetKey] = useState(0);
  const [stats, setStats] = useState<Stats | null>(null);
  const [compareStats, setCompareStats] = useState<Stats | null>(null);
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
  const [focusKey, setFocusKey] = useState(0);
  const [error, setError] = useState("");
  const [statsError, setStatsError] = useState("");
  const [loading, setLoading] = useState(true);
  const [importing, setImporting] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const data = cases.find((c) => c.id === caseId);
  const model = models.find((m) => m.id === modelId);
  const caseModelReady =
    !!model?.available && !!data && (modelId.includes("demo") ? data.demo :
      data.label_preset === "mu_glioma_post" && ["t1n", "t1c", "t2w", "t2f"].every(m => data.modalities.includes(m)));
  // Keep the monitor alive through terminal-result hydration and paused recovery.
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
      setCaseId(running?.case_id || c.cases[0]?.id || "");
      setModelId(
        running?.model_id ||
          m.models.find((x) => x.available)?.id ||
          m.models[0]?.id ||
          "",
      );
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
    setModality(data.modalities.includes("t1c") ? "t1c" : data.modalities[0]);
    setSegmentation(data.segmentations[0]?.id || "");
    setComparison(data.segmentations[1]?.id || data.segmentations[0]?.id || "");
    setVisibleLabels(data.labels.map((l) => l.id));
    setStats(null);
    setCompareStats(null);
    setIsolated(false);
    setExploded(false);
    setFocusKey(0);
    setResetKey((v) => v + 1);
  }, [caseId]); // case transitions, not job refreshes
  useEffect(() => {
    if (!caseId) return;
    const controller = new AbortController();
    setStatsError("");
    setStats(null);
    setCompareStats(null);
    Promise.all([
      api<Stats>(
        `/cases/${caseId}/stats?segmentation=${encodeURIComponent(segmentation)}&reference=${encodeURIComponent(segmentation)}`,
        { signal: controller.signal },
      ),
      api<Stats>(
        `/cases/${caseId}/stats?segmentation=${encodeURIComponent(comparison)}&reference=${encodeURIComponent(segmentation)}`,
        { signal: controller.signal },
      ),
    ])
      .then(([a, b]) => {
        if (controller.signal.aborted) return;
        setStats(a);
        setCompareStats(b);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setStatsError(e.message);
      });
    return () => controller.abort();
  }, [caseId, segmentation, comparison]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 5000);
    return () => clearTimeout(timer);
  }, [notice]);
  useEffect(() => {
    if (!monitoredJobId || jobConnection?.kind === "missing" || jobConnection?.kind === "paused") return;
    return monitorInferenceJob({
      jobId: monitoredJobId,
      isCurrent: () =>
        selectedCaseId.current === caseId && monitoredId.current === monitoredJobId,
      onJob: setActiveJob,
      onConnection: setJobConnection,
      onFinished: (job, allJobs, updated) => {
        setJobs(allJobs);
        setMonitoredJobId(null);
        if (updated) {
          setCases((v) => v.map((c) => (c.id === updated.id ? updated : c)));
          if (job.case_id === caseId && job.segmentation_id) {
            setSegmentation((current) =>
              updated.segmentations.some((s) => s.id === current)
                ? current
                : job.segmentation_id!,
            );
            setComparison(job.segmentation_id);
            setTab(updated.segmentations.length > 1 ? "compare" : "workspace");
          }
          setNotice("추론이 완료되었습니다. 결과를 확인할 수 있습니다.");
        } else if (["failed", "error"].includes(job.status)) {
          setError(job.message || "추론에 실패했습니다.");
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
      setCaseId((current) => caseResult.cases.some((c) => c.id === current)
        ? current : caseResult.cases[0]?.id || "");
      const job = jobResult.jobs.find((j) => j.id === monitoredId.current) ||
        jobResult.jobs.find((j) => j.case_id === selectedCaseId.current && isActiveJob(j)) ||
        jobResult.jobs.find(isActiveJob);
      setActiveJob(job || null);
      setMonitoredJobId(job?.id || null);
      setJobConnection(null);
      setMonitorRevision((v) => v + 1);
      if (!job) setNotice("작업 목록과 저장된 마스크를 다시 불러왔습니다. 현재 실행 중인 작업은 없습니다.");
    } catch (e) {
      if (!controller.signal.aborted) setJobConnection({
        kind: "paused",
        message: `다시 연결하지 못했습니다. ${(e as Error).message}`,
      });
    } finally {
      if (!controller.signal.aborted) setReconnecting(false);
    }
  };
  const toggleLabel = (id: number) =>
    setVisibleLabels((v) =>
      v.includes(id) ? v.filter((n) => n !== id) : [...v, id],
    );
  const reset = () => {
    setIsolated(false);
    setExploded(false);
    setBrainOpacity(0.13);
    setOpacity(0.6);
    setContrast(100);
    setOverlay(true);
    setVisibleLabels(data?.labels.map((l) => l.id) || []);
    setResetKey((v) => v + 1);
  };
  const startJob = async () => {
    if (!data || !caseModelReady || active) return;
    setSubmitting(true);
    setError("");
    try {
      const job = await api<Job>("/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ case_id: caseId, model_id: modelId }),
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
    setTab("workspace");
    setNotice(`${c.name} 볼륨을 가져왔습니다.`);
  };
  const meshProps = useMemo(
    () =>
      data
        ? {
            caseId: data.id,
            labels: data.labels,
            visibleLabels,
            isolated,
            brainOpacity,
            exploded,
            resetKey,
          }
        : null,
    [data, visibleLabels, isolated, brainOpacity, exploded, resetKey],
  );
  const selectedName =
    data?.segmentations.find((s) => s.id === segmentation)?.name ||
    "마스크 없음";
  const predictedName =
    data?.segmentations.find((s) => s.id === comparison)?.name || "결과 없음";
  const exportResult = () => {
    if (segmentation)
      window.location.assign(
        `/api/cases/${caseId}/segmentations/${encodeURIComponent(segmentation)}/download`,
      );
  };
  const currentJobs = jobs.filter((j) => j.case_id === caseId);

  return (
    <div className={`app-shell ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <header className="app-header">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setTab("workspace");
          }}
        >
          <span className="brand-mark">
            <Scan size={24} />
          </span>
          <span>
            neuro<span className="brand-slash">/</span>
            <b>lab</b>
            <small>VOLUMETRIC WORKSPACE</small>
          </span>
        </a>
        <nav className="top-nav" aria-label="주 메뉴">
          {tabs.map((t) => (
            <button
              key={t.id}
              className={tab === t.id ? "selected" : ""}
              onClick={() => setTab(t.id)}
            >
              <t.icon size={16} />
              <span>{t.name}</span>
            </button>
          ))}
        </nav>
        <div className="header-status">
          <span className="live-dot" />
          로컬 워크스페이스<span className="version">MVP 0.1</span>
        </div>
      </header>
      <aside className="sidebar">
        <div className="sidebar-title">
          <span>EXPLORER</span>
          <button
            className="icon-btn"
            aria-label="사이드바 접기"
            onClick={() => setSidebarOpen(false)}
          >
            <PanelLeftClose size={16} />
          </button>
        </div>
        <div className="section-label">
          현재 사례{" "}
          <button
            className="icon-btn small"
            aria-label="새 MRI 사례 가져오기"
            onClick={() => setImporting(true)}
          >
            <Plus size={16} />
          </button>
        </div>
        {data ? (
          <>
            <div className="case-select-wrap">
              <select
                aria-label="MRI 사례 선택"
                value={caseId}
                onChange={(e) => setCaseId(e.target.value)}
              >
                {cases.map((c) => (
                  <option value={c.id} key={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
              <ChevronDown size={15} />
            </div>
            <div className="case-info">
              <span className={`source-badge ${data.demo ? "demo" : ""}`}>
                {data.demo ? <Sparkles size={12} /> : <Files size={12} />}{" "}
                {data.demo ? "합성 데모" : "NIfTI 볼륨"}
              </span>
              <p>
                {data.demo ? "조작 연습용 가상 데이터" : "로컬에서 가져온 MRI"}
              </p>
              <dl>
                <div>
                  <dt>크기</dt>
                  <dd>{data.shape.join(" × ")}</dd>
                </div>
                <div>
                  <dt>복셀 간격</dt>
                  <dd>
                    {data.spacing.map((v) => number(v, 1)).join(" × ")} mm
                  </dd>
                </div>
                <div>
                  <dt>좌표계</dt>
                  <dd>{data.orientation || "RAS"}</dd>
                </div>
              </dl>
            </div>
            <div className="sidebar-divider" />
            <div className="section-label">MRI 시퀀스</div>
            <div className="modality-options">
              {data.modalities.map((m) => (
                <button
                  key={m}
                  className={m === modality ? "chosen" : ""}
                  onClick={() => setModality(m)}
                >
                  {modalityName[m] || m}
                </button>
              ))}
            </div>
            <div className="sidebar-divider" />
            <div className="section-label">
              추론 모델 <Cpu size={15} />
            </div>
            <div className="model-select-wrap">
              <select
                aria-label="추론 모델 선택"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                    {m.available ? "" : " · 미연결"}
                  </option>
                ))}
              </select>
            </div>
            <p className="model-description">{model?.description}</p>
            <div className={`model-status ${caseModelReady ? "ready" : ""}`}>
              <span className="live-dot" />
              {caseModelReady
                ? "실행 가능"
                : model?.available
                  ? "이 사례에서 실행 불가"
                  : "학습 가중치 연결 필요"}
            </div>
            {!model?.available && (
              <p className="model-reason">{model?.reason}</p>
            )}
            <button
              className="button primary run-button"
              disabled={
                !caseModelReady || !!active || submitting
              }
              onClick={startJob}
            >
              {active || submitting ? (
                <LoaderCircle className="spin" size={16} />
              ) : (
                <Play size={16} />
              )}{" "}
              {active
                ? jobConnection
                  ? jobConnection.kind === "retrying" ? "상태 다시 확인 중" : "상태 확인 필요"
                  : activeJob && isCompletedJob(activeJob) ? "결과 불러오는 중" : "추론 실행 중"
                : modelId.includes("demo")
                  ? "데모 파이프라인 실행"
                  : "3D 분할 실행"}
            </button>
            {!data.demo && modelId.includes("demo") && (
              <p className="model-reason">
                합성 데모는 실제 영상에 적용할 수 없습니다.
              </p>
            )}
            {activeJob && activeJob.case_id === caseId && (
              <div className="job-progress" aria-live="polite">
                <div>
                  <span>
                    {activeJob.status === "completed"
                      ? "완료"
                      : activeJob.message}
                  </span>
                  <b>{Math.round(activeJob.progress)}%</b>
                </div>
                <progress max={100} value={activeJob.progress} />
                <small>
                  {activeJob.elapsed_seconds != null
                    ? `${number(activeJob.elapsed_seconds, 1)}초`
                    : "작업 상태를 확인하고 있습니다"}
                </small>
              </div>
            )}
          </>
        ) : (
          <p className="muted">사례를 불러오는 중입니다.</p>
        )}
        <div className="sidebar-bottom">
          <BookOpen size={18} />
          <div>
            <strong>3D 모델, 직접 이해하기</strong>
            <button onClick={() => setTab("learn")}>
              학습 가이드 열기 <ArrowRight size={13} />
            </button>
          </div>
        </div>
      </aside>
      <main className="main-content">
        <div className="workspace-heading">
          <div>
            <div className="breadcrumb">
              {!sidebarOpen && (
                <button
                  className="icon-btn"
                  aria-label="사이드바 열기"
                  onClick={() => setSidebarOpen(true)}
                >
                  <PanelLeftClose size={15} />
                </button>
              )}
              RESEARCH <span>/</span> BRAIN MRI <span>/</span> 3D
            </div>
            <h1>
              {tab === "workspace"
                ? "3D 종양 워크스페이스"
                : tab === "compare"
                  ? "모델 결과 비교"
                  : "3D 모델 실험실"}
            </h1>
          </div>
          <button
            className="button secondary import-button"
            onClick={() => setImporting(true)}
          >
            <Upload size={16} /> NIfTI 가져오기
          </button>
        </div>
        {error && (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            <div>
              {!data && <button onClick={initialize}>다시 연결</button>}
              <button
                className="icon-btn"
                aria-label="오류 알림 닫기"
                onClick={() => setError("")}
              >
                <X size={16} />
              </button>
            </div>
          </div>
        )}
        {jobConnection && (
          <div className="error-banner" role="status">
            <span>{jobConnection.message}</span>
            {jobConnection.kind !== "retrying" && (
              <button onClick={reconnectJobs} disabled={reconnecting}>
                {reconnecting ? "다시 연결 중…" : "작업·결과 다시 연결"}
              </button>
            )}
          </div>
        )}
        {loading && (
          <div className="loading-workspace">
            <LoaderCircle className="spin" size={30} />
            <h2>워크스페이스 준비 중</h2>
            <p>데모 볼륨과 모델 상태를 확인하고 있습니다.</p>
          </div>
        )}
        {!loading && !data && (
          <div className="empty-workspace">
            <Files size={40} />
            <h2>MRI 볼륨을 추가하세요</h2>
            <p>서버가 실행 중인지 확인한 후 사례를 가져오세요.</p>
            <button
              className="button primary"
              onClick={() => setImporting(true)}
            >
              NIfTI 가져오기
            </button>
          </div>
        )}
        {data && meshProps && tab !== "learn" && (
          <>
            <div className="workspace-toolbar">
              <div className="segmentation-picker">
                <Layers3 size={16} />
                <label htmlFor="segmentation-select">
                  {tab === "compare" ? "기준 결과" : "분할 결과"}
                </label>
                <select
                  id="segmentation-select"
                  value={segmentation}
                  onChange={(e) => setSegmentation(e.target.value)}
                >
                  {!data.segmentations.length && (
                    <option value="">마스크 없음</option>
                  )}
                  {data.segmentations.map((s) => (
                    <option value={s.id} key={s.id}>
                      {s.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="toolbar-actions">
                {tab === "workspace" && (
                  <button
                    className="icon-btn"
                    title="세 단면을 종양 중심으로 이동"
                    aria-label="종양 위치 찾기"
                    disabled={!stats?.tumor_center_voxel}
                    onClick={() => setFocusKey((v) => v + 1)}
                  >
                    <Scan size={17} />
                  </button>
                )}
                <button
                  className={`pill-button ${isolated ? "selected" : ""}`}
                  aria-pressed={isolated}
                  onClick={() => setIsolated((v) => !v)}
                >
                  <Focus size={15} /> 종양만 보기
                </button>
                <button
                  className="icon-btn"
                  title="모든 보기 설정 초기화"
                  aria-label="모든 보기 설정 초기화"
                  onClick={reset}
                >
                  <RotateCcw size={16} />
                </button>
              </div>
            </div>
            {data.demo && (
              <div className="demo-notice">
                <Sparkles size={14} />
                <span>
                  <strong>합성 데모</strong> · 화면의 영상과 분할 결과는 조작
                  연습용이며, MU-Glioma-Post 실제 영상이나 학습된 모델의 예측이 아닙니다.
                </span>
              </div>
            )}
            {tab === "workspace" ? (
              <div className="viewer-layout">
                <section className="primary-view">
                  <MeshViewer {...meshProps} segmentation={segmentation} />
                  <div className="viewer-control-bar">
                    <div>
                      <button
                        className={`view-mode ${!isolated ? "selected" : ""}`}
                        onClick={() => setIsolated(false)}
                      >
                        <Box size={15} /> 뇌 + 종양
                      </button>
                      <button
                        className={`view-mode ${isolated ? "selected" : ""}`}
                        onClick={() => setIsolated(true)}
                      >
                        <Focus size={15} /> 종양만
                      </button>
                    </div>
                    <div className="explode-control">
                      <label htmlFor="explode">영역 펼쳐 보기</label>
                      <Toggle
                        checked={exploded}
                        label="종양 영역 펼쳐 보기"
                        onChange={() => setExploded((v) => !v)}
                      />
                    </div>
                  </div>
                  {exploded && (
                    <div className="exploded-note">
                      영역 비교를 위해 위치를 이동한 보기입니다. 실제 해부학적
                      위치는 ‘영역 펼쳐 보기’를 끄면 복원됩니다.
                    </div>
                  )}
                </section>
                <div className="slice-stack">
                  {(["axial", "coronal", "sagittal"] as const).map((plane) => (
                    <SliceViewer
                      key={plane}
                      data={data}
                      plane={plane}
                      modality={modality}
                      segmentation={segmentation}
                      visibleLabels={visibleLabels}
                      overlay={overlay}
                      opacity={opacity}
                      window={contrast}
                      resetKey={resetKey}
                      focusKey={focusKey}
                      focusVoxel={stats?.tumor_center_voxel}
                    />
                  ))}
                </div>
              </div>
            ) : (
              <div className="comparison-layout">
                <section>
                  <div className="comparison-title">
                    <span className="tiny-tag">REFERENCE</span>
                    <strong>{selectedName}</strong>
                  </div>
                  <MeshViewer
                    {...meshProps}
                    segmentation={segmentation}
                    compact
                  />
                </section>
                <section>
                  <div className="comparison-title">
                    <span className="tiny-tag mint">COMPARE</span>
                    <select
                      aria-label="비교할 분할 결과"
                      value={comparison}
                      onChange={(e) => setComparison(e.target.value)}
                    >
                      {data.segmentations.map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <MeshViewer
                    {...meshProps}
                    segmentation={comparison}
                    compact
                  />
                </section>
              </div>
            )}
            {tab === "compare" && (
              <div className="comparison-metrics">
                <div>
                  <span>영역 겹침 · Dice</span>
                  <strong>{number(compareStats?.dice, 3)}</strong>
                  <small>1에 가까울수록 일치</small>
                </div>
                <div>
                  <span>경계 거리 · HD95</span>
                  <strong>
                    {number(compareStats?.hd95_mm)} <em>mm</em>
                  </strong>
                  <small>0에 가까울수록 일치</small>
                </div>
                <div>
                  <span>전체 라벨 부피 차이</span>
                  <strong>
                    {stats?.total_volume_ml != null &&
                    compareStats?.total_volume_ml != null
                      ? `${compareStats.total_volume_ml - stats.total_volume_ml > 0 ? "+" : ""}${number(compareStats.total_volume_ml - stats.total_volume_ml)}`
                      : "—"}{" "}
                    <em>mL</em>
                  </strong>
                  <small>
                    {predictedName} − {selectedName}
                  </small>
                </div>
                <p>
                  {comparison === segmentation
                    ? "같은 결과를 선택했습니다. 서로 다른 결과를 선택하면 차이를 확인할 수 있습니다."
                    : "현재 선택한 두 마스크의 전체 전경 기준 지표입니다. 두 예측의 일치만으로 정확도를 판단할 수는 없습니다."}
                </p>
              </div>
            )}
            <div className="bottom-layout">
              <section className="panel region-panel">
                <header className="panel-heading">
                  <div>
                    <Layers3 size={17} />
                    <h2>분할 영역</h2>
                    <span className="count-tag">{data.labels.length}</span>
                  </div>
                  <button
                    className="text-button"
                    onClick={() =>
                      setVisibleLabels(
                        visibleLabels.length === data.labels.length
                          ? []
                          : data.labels.map((l) => l.id),
                      )
                    }
                  >
                    {visibleLabels.length === data.labels.length
                      ? "모두 숨기기"
                      : "모두 보기"}
                  </button>
                </header>
                <div className="region-summary">
                  <div>
                    <span>전체 라벨 부피</span>
                    <strong>
                      {number(stats?.total_volume_ml)}
                      <small> mL</small>
                    </strong>
                  </div>
                  <div className="volume-bar">
                    {stats?.regions.map((r) => (
                      <span
                        key={r.label}
                        style={{
                          background: r.color,
                          flex: r.volume_ml || 0.001,
                        }}
                        title={`${r.name}: ${number(r.volume_ml)} mL`}
                      />
                    ))}
                  </div>
                </div>
                {statsError && (
                  <div className="inline-error" role="alert">
                    {statsError}
                  </div>
                )}
                <div className="region-table">
                  <div className="region-table-head">
                    <span>영역 / 라벨</span>
                    <span>부피 (mL)</span>
                    <span>{tab === "compare" ? "비교 (mL)" : "연결 성분"}</span>
                    <span>표시</span>
                  </div>
                  {data.labels.map((l) => {
                    const region = stats?.regions.find((r) => r.label === l.id);
                    const other = compareStats?.regions.find(
                      (r) => r.label === l.id,
                    );
                    return (
                      <div
                        key={l.id}
                        className={`region-row ${visibleLabels.includes(l.id) ? "" : "dimmed"}`}
                      >
                        <div>
                          <span
                            className="region-dot"
                            style={{ background: l.color }}
                          />
                          <span>
                            {l.name}
                            <small>LABEL {l.id}</small>
                          </span>
                        </div>
                        <span className="mono">
                          {number(region?.volume_ml)}
                        </span>
                        <span className="mono muted">
                          {tab === "compare"
                            ? number(other?.volume_ml)
                            : (region?.components ?? "—")}
                        </span>
                        <button
                          className="icon-btn"
                          aria-label={`${l.name} 영역 ${visibleLabels.includes(l.id) ? "숨기기" : "표시"}`}
                          aria-pressed={visibleLabels.includes(l.id)}
                          onClick={() => toggleLabel(l.id)}
                        >
                          {visibleLabels.includes(l.id) ? (
                            <Eye size={16} />
                          ) : (
                            <EyeOff size={16} />
                          )}
                        </button>
                      </div>
                    );
                  })}
                </div>
                <div className="region-footnote">
                  부피는 복셀의 실제 크기를 반영합니다. 수술강(RC) 등 비종양
                  영역이 포함될 수 있습니다.
                </div>
              </section>
              <section className="panel settings-panel">
                <header className="panel-heading">
                  <div>
                    <Settings2 size={17} />
                    <h2>보기 설정</h2>
                  </div>
                  <button
                    className="icon-btn"
                    aria-label="보기 설정 리셋"
                    onClick={reset}
                  >
                    <RotateCcw size={14} />
                  </button>
                </header>
                <div className="setting-row">
                  <span>단면에 마스크 표시</span>
                  <Toggle
                    checked={overlay}
                    label="단면에 마스크 표시"
                    onChange={() => setOverlay((v) => !v)}
                  />
                </div>
                <label className="range-setting">
                  <span>
                    마스크 불투명도 <b>{Math.round(opacity * 100)}%</b>
                  </span>
                  <input
                    aria-label="마스크 불투명도"
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={opacity}
                    onChange={(e) => setOpacity(+e.target.value)}
                  />
                </label>
                <label className="range-setting">
                  <span>
                    뇌 불투명도 <b>{Math.round(brainOpacity * 100)}%</b>
                  </span>
                  <input
                    aria-label="뇌 불투명도"
                    type="range"
                    min={0.02}
                    max={0.8}
                    step={0.01}
                    value={brainOpacity}
                    disabled={isolated}
                    onChange={(e) => setBrainOpacity(+e.target.value)}
                  />
                </label>
                <label className="range-setting">
                  <span>
                    영상 윈도우 <b>{contrast}%</b>
                  </span>
                  <input
                    aria-label="영상 윈도우"
                    type="range"
                    min={25}
                    max={180}
                    step={5}
                    value={contrast}
                    onChange={(e) => setContrast(+e.target.value)}
                  />
                </label>
                <button
                  className="button secondary export-button"
                  onClick={exportResult}
                  disabled={!segmentation}
                >
                  <ArrowDownToLine size={16} /> 선택한 마스크 저장{" "}
                  <small>.nii.gz</small>
                </button>
              </section>
            </div>
          </>
        )}
        {tab === "learn" && (
          <div className="learning-content">
            <div className="learning-intro">
              <div className="eyebrow">FROM SLICES TO VOLUMES</div>
              <h2>
                한 장의 이미지에서,
                <br />
                <span>공간을 이해하는 모델로.</span>
              </h2>
              <p>
                3D U-Net으로 기본 구조를 익히고, 동일한 데이터에서 학습 방법과
                아키텍처를 비교하세요.
              </p>
              <span className="learning-badge">
                <BookOpen size={15} /> 실행 방법과 설정은 프로젝트 README에
                정리되어 있습니다.
              </span>
            </div>
            <div className="learning-cards">
              {[
                {
                  n: "01",
                  title: "3D U-Net",
                  tag: "기본 구조",
                  body: "3D 합성곱과 skip connection으로 볼륨 전체의 문맥을 학습합니다. 패치 크기, 손실함수, 시퀀스 구성을 바꾸며 실험하세요.",
                  link: "https://arxiv.org/abs/1606.06650",
                },
                {
                  n: "02",
                  title: "nnU-Net v2",
                  tag: "학습 방법론",
                  body: "데이터의 특성에 맞게 전처리와 학습 구성을 정하는 프레임워크입니다. 직접 구현한 모델을 비교할 기준을 만듭니다.",
                  link: "https://github.com/MIC-DKFZ/nnUNet",
                },
                {
                  n: "03",
                  title: "Swin UNETR",
                  tag: "아키텍처 확장",
                  body: "Swin Transformer 인코더를 사용하는 3D 분할 모델입니다. CNN과의 정확도·시간·메모리 차이를 확인하세요.",
                  link: "https://arxiv.org/abs/2201.01266",
                },
              ].map((card) => (
                <article className="learning-card" key={card.n}>
                  <span className="step-number">{card.n}</span>
                  <span className="tiny-tag">{card.tag}</span>
                  <h3>{card.title}</h3>
                  <p>{card.body}</p>
                  <a href={card.link} target="_blank" rel="noreferrer">
                    논문 · 공식 자료 <ArrowRight size={15} />
                  </a>
                </article>
              ))}
            </div>
            <section className="panel">
              <header className="panel-heading">
                <div>
                  <Cpu size={17} />
                  <h2>모델 연결 상태</h2>
                </div>
              </header>
              <div className="model-registry">
                {models.map((m) => (
                  <div key={m.id}>
                    <div>
                      <strong>{m.name}</strong>
                      <p>{m.description}</p>
                    </div>
                    <span
                      className={`status-chip ${m.available ? "ready" : ""}`}
                    >
                      {m.available ? (
                        <CheckCircle2 size={14} />
                      ) : (
                        <Cpu size={14} />
                      )}{" "}
                      {m.available ? "사용 가능" : "가중치 미연결"}
                    </span>
                  </div>
                ))}
              </div>
            </section>
            <section className="panel">
              <header className="panel-heading">
                <div>
                  <Activity size={17} />
                  <h2>현재 사례의 추론 기록</h2>
                </div>
                <button
                  className="text-button"
                  onClick={() =>
                    api<{ jobs: Job[] }>("/jobs")
                      .then((r) => setJobs(r.jobs))
                      .catch((e) => setError(e.message))
                  }
                >
                  새로고침
                </button>
              </header>
              {currentJobs.length ? (
                <div className="job-history">
                  {currentJobs.map((j) => (
                    <div key={j.id}>
                      <Cpu size={18} />
                      <div>
                        <strong>
                          {models.find((m) => m.id === j.model_id)?.name ||
                            j.model_id}
                        </strong>
                        <small>{j.message}</small>
                      </div>
                      <span className="status-chip">{j.status}</span>
                      <span className="mono">
                        {number(j.elapsed_seconds, 1)} s
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-history">
                  <Activity size={24} />
                  <p>아직 추론 기록이 없습니다.</p>
                  <span>
                    왼쪽에서 모델을 선택하고 파이프라인을 실행해 보세요.
                  </span>
                </div>
              )}
            </section>
            <div className="study-note">
              <CheckCircle2 size={18} />
              <p>
                공정한 비교를 위해 환자 단위 분할과 평가 조건을 동일하게
                유지하세요. 학습은 로컬 CLI에서 실행하며, 이 화면은 모델 상태와
                추론 기록을 보여줍니다.
              </p>
            </div>
          </div>
        )}
        <footer className="workspace-footer">
          <span>
            <ShieldCheck size={13} /> 연구·학습용 워크스페이스
          </span>
          <span>
            <a href="https://www.cancerimagingarchive.net/collection/mu-glioma-post/" target="_blank" rel="noreferrer">
              MU-Glioma-Post
            </a>{" "}· NIfTI · 3D segmentation
          </span>
        </footer>
      </main>
      {importing && (
        <ImportDialog
          onClose={() => setImporting(false)}
          onImported={imported}
        />
      )}
      {notice && (
        <div className="toast" role="status">
          <CheckCircle2 size={18} />
          {notice}
          <button
            className="icon-btn"
            aria-label="알림 닫기"
            onClick={() => setNotice("")}
          >
            <X size={15} />
          </button>
        </div>
      )}
    </div>
  );
}
