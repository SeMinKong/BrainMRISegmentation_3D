import { AlertTriangle, ChevronLeft, ChevronRight, LoaderCircle, Play, RotateCcw } from "lucide-react";
import { number } from "../api";
import type { Case, Job, Model, Segmentation } from "../api";
import { isCompletedJob } from "../inferenceJob";
import type { JobConnection } from "../inferenceJob";
import type { Patient } from "../patients";
import { Help } from "./Help";

type Props = {
  data: Case;
  patient?: Patient;
  model: Model | undefined;
  ready: boolean;
  prediction: Segmentation | null;
  active: boolean;
  submitting: boolean;
  activeJob: Job | null;
  jobConnection: JobConnection;
  /** The last job on this case that failed, until the user retries or moves on. */
  failure?: Job | null;
  onPredict: () => void;
  onDismissFailure?: () => void;
  /** Step through the visits in the order the list shows them. */
  position?: { index: number; total: number };
  onPrevious?: () => void;
  onNext?: () => void;
};

/** The server's technical progress messages as three stages a non-clinician can follow. */
export function stageOf(job: Job): { stage: number; name: string } {
  if (isCompletedJob(job)) return { stage: 3, name: "완료" };
  if (job.progress >= 90) return { stage: 3, name: "결과 저장" };
  if (job.progress >= 20 || /infer|slid|window|predict/i.test(job.message)) return { stage: 2, name: "모델 계산" };
  return { stage: 1, name: "영상 준비" };
}
const STAGES = ["영상 준비", "모델 계산", "결과 저장"];

/** One card that says which scan this is, whether the model saw it in training, and lets you run the model. */
export default function CaseHeader(props: Props) {
  const { data, patient, model, ready, prediction, active, submitting, activeJob, jobConnection, failure, onPredict, onDismissFailure, position, onPrevious, onNext } = props;
  const timepoint = patient?.timepoints.find((t) => t.case.id === data.id);
  const split = data.study?.split;
  const running = active || submitting;
  const runLabel = running
    ? jobConnection
      ? jobConnection.kind === "retrying" ? "상태 다시 확인 중" : "상태 확인 필요"
      : activeJob && isCompletedJob(activeJob) ? "결과 불러오는 중" : "예측 중…"
    : prediction ? "다시 예측하기" : "내 모델로 예측하기";
  const blocked = !ready && !running
    ? !model?.available
      ? "모델 체크포인트가 연결되지 않았습니다. .env의 MRI_UNET_CHECKPOINT를 확인하세요."
      : data.demo
        ? "합성 데모 영상에는 학습한 모델을 적용하지 않습니다."
        : "이 검사에는 MRI 네 장(T1·T1 조영·T2·FLAIR)이 모두 있어야 예측할 수 있습니다."
    : "";
  const stage = activeJob && running ? stageOf(activeJob) : null;

  return (
    <section className="neu-card case-header" aria-label="현재 검사">
      <div className="case-title">
        <div className="case-nav">
          {position && (
            <button className="neu-icon" onClick={onPrevious} disabled={!onPrevious} aria-label="이전 검사" title="이전 검사 (←)" aria-keyshortcuts="ArrowLeft">
              <ChevronLeft size={18} />
            </button>
          )}
          <div>
            <h2>{patient?.name ?? data.name}{timepoint?.index != null ? ` · ${timepoint.label} 검사` : ""}</h2>
            <div className="badges">
              {position && <span className="badge mono" title="목록에서의 위치">{position.index + 1} / {position.total}</span>}
              {split === "val" && (
                <span className="badge unseen">
                  학습에 쓰지 않은 검사 <Help term="unseen" />
                </span>
              )}
              {split === "train" && (
                <span className="badge seen">
                  학습에 쓴 검사 · 참고용 <Help term="unseen" />
                </span>
              )}
              {data.demo && <span className="badge">합성 데모</span>}
              {!split && !data.demo && <span className="badge">가져온 MRI</span>}
              {prediction && <span className="badge done">예측 완료</span>}
            </div>
          </div>
          {position && (
            <button className="neu-icon" onClick={onNext} disabled={!onNext} aria-label="다음 검사" title="다음 검사 (→)" aria-keyshortcuts="ArrowRight">
              <ChevronRight size={18} />
            </button>
          )}
        </div>
        <div className="case-actions">
          <button className={`neu-btn primary big ${running ? "busy" : ""}`} disabled={!ready || running} onClick={onPredict} title={blocked || undefined}>
            {running ? <LoaderCircle className="spin" size={18} /> : prediction ? <RotateCcw size={18} /> : <Play size={18} />} {runLabel}
          </button>
          {model && (
            <span className="model-note muted">
              {model.name} <Help term="model" />
              {model.training?.validation_mean_dice != null && (
                <>
                  {" · "}학습에 쓰지 않은 {model.training.val_cases ?? "—"}개 검사 평균 일치도 <b>{number(model.training.validation_mean_dice, 2)}</b>
                </>
              )}
            </span>
          )}
        </div>
      </div>
      {blocked && <p className="blocked-note">{blocked}</p>}
      {activeJob && activeJob.case_id === data.id && running && stage && (
        <div className="job-progress" aria-live="polite">
          <ol className="stages" aria-label="진행 단계">
            {STAGES.map((name, i) => (
              <li key={name} className={i + 1 < stage.stage ? "done" : i + 1 === stage.stage ? "now" : ""}>
                <span className="stage-dot">{i + 1 < stage.stage ? "✓" : i + 1}</span> {name}
              </li>
            ))}
          </ol>
          <progress max={100} value={activeJob.progress} aria-label={`${stage.name} ${Math.round(activeJob.progress)}%`} />
          <small>
            {activeJob.elapsed_seconds != null ? `${number(activeJob.elapsed_seconds, 1)}초 경과 · 보통 10초 안에 끝납니다` : "작업 상태를 확인하고 있습니다"}
            {activeJob.status === "queued" && " · 앞선 작업이 끝나면 시작합니다"}
          </small>
        </div>
      )}
      {failure && !running && (
        <div className="failure-card" role="alert">
          <AlertTriangle size={18} />
          <div>
            <strong>예측이 실패했습니다.</strong>
            <p>{failure.message}</p>
            <p className="muted small-print">흔한 원인: 체크포인트 경로가 바뀜, GPU 메모리 부족, MRI 네 장 중 하나가 손상됨. 서버 로그(runs/server-8000.err)에 자세한 내용이 남습니다.</p>
          </div>
          <div className="failure-actions">
            <button className="neu-btn" onClick={onPredict} disabled={!ready}>다시 시도</button>
            {onDismissFailure && <button className="neu-btn" onClick={onDismissFailure}>닫기</button>}
          </div>
        </div>
      )}
    </section>
  );
}
