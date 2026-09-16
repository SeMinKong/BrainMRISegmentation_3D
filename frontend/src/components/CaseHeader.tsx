import { LoaderCircle, Play, RotateCcw } from "lucide-react";
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
  onPredict: () => void;
};

/** One card that says which scan this is, whether the model saw it in training, and lets you run the model. */
export default function CaseHeader({ data, patient, model, ready, prediction, active, submitting, activeJob, jobConnection, onPredict }: Props) {
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

  return (
    <section className="neu-card case-header" aria-label="현재 검사">
      <div className="case-title">
        <div>
          <h2>{patient?.name ?? data.name}{timepoint?.index != null ? ` · ${timepoint.label} 검사` : ""}</h2>
          <div className="badges">
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
      {activeJob && activeJob.case_id === data.id && running && (
        <div className="job-progress" aria-live="polite">
          <div>
            <span>{activeJob.status === "completed" ? "완료" : activeJob.message}</span>
            <b>{Math.round(activeJob.progress)}%</b>
          </div>
          <progress max={100} value={activeJob.progress} />
          <small>{activeJob.elapsed_seconds != null ? `${number(activeJob.elapsed_seconds, 1)}초 경과 · 보통 10초 안에 끝납니다` : "작업 상태를 확인하고 있습니다"}</small>
        </div>
      )}
    </section>
  );
}
