import { CheckCircle2, Circle, LoaderCircle, PlayCircle, X } from "lucide-react";
import { number } from "../api";
import type { Case, Model, Overview as OverviewData } from "../api";
import { LABEL_GUIDE } from "../labels";
import { patientName } from "../patients";
import { Help } from "./Help";
import { verdict, verdictClass } from "./ResultCard";

export type BatchProgress = { total: number; done: number; failed: number; running: boolean };

type Props = {
  overview: OverviewData | null;
  loading: boolean;
  model: Model | undefined;
  cases: Case[];
  batch: BatchProgress | null;
  onBatch: () => void;
  onOpenCase: (caseId: string) => void;
  /** First-visit checklist; hidden once dismissed (remembered in this browser). */
  showChecklist: boolean;
  onDismissChecklist: () => void;
};

const WORST = 10;

const visitName = (caseId: string, name: string) => {
  const match = /^(PatientID_\d+)_Timepoint_(\d+)$/i.exec(caseId);
  return match ? `${patientName(match[1])} · ${match[2]}차` : name;
};

/**
 * First screen: how the model does on every validation scan it has not trained on. One number, its
 * distribution, each region, and the visits to look at first.
 */
export default function Overview({ overview, loading, model, cases, batch, onBatch, onOpenCase, showChecklist, onDismissChecklist }: Props) {
  const connected = !!model?.available && !model.demo_only;
  const candidates = cases.filter((c) => c.study?.split === "val" && !c.demo && c.segmentations.some((s) => s.id === "reference"));
  const predicted = overview?.predicted ?? 0;
  const remaining = Math.max(0, (overview?.candidates ?? candidates.length) - predicted);
  const summary = overview?.summary;
  const scored = (overview?.cases ?? []).filter((c) => c.dice != null);
  const worst = scored.slice(0, WORST);
  const maxCount = Math.max(1, ...(summary?.histogram ?? []).map((b) => b.count));
  const labels = ["1", "2", "3", "4"].map((id) => ({ id, ...LABEL_GUIDE[Number(id)], color: cases[0]?.labels.find((l) => l.id === Number(id))?.color ?? "#888", stat: summary?.labels[id] }));
  const running = !!batch?.running;

  return (
    <div className="overview">
      {showChecklist && (
        <section className="neu-card checklist" aria-label="시작하기">
          <header className="card-head">
            <div>
              <h3>시작하기</h3>
              <p>세 단계면 내 모델의 실력을 확인할 수 있습니다.</p>
            </div>
            <button className="neu-icon" aria-label="안내 닫기" onClick={onDismissChecklist}><X size={15} /></button>
          </header>
          <ol className="steps">
            <li className={connected ? "done" : ""}>
              {connected ? <CheckCircle2 size={18} /> : <Circle size={18} />}
              <div>
                <strong>모델 연결</strong>
                <span>{connected ? `${model?.name} 체크포인트가 연결되어 있습니다.` : ".env 파일의 MRI_UNET_CHECKPOINT에 학습한 best.pt 경로를 적고 서버를 다시 시작하세요."}</span>
              </div>
            </li>
            <li className={predicted > 0 ? "done" : ""}>
              {predicted > 0 ? <CheckCircle2 size={18} /> : <Circle size={18} />}
              <div>
                <strong>학습에 쓰지 않은 검사 전체 예측</strong>
                <span>{predicted > 0 ? `${predicted}개 검사에 예측이 있습니다.` : "아래 버튼 하나로 검증 검사 전체를 순서대로 예측합니다. 검사당 10초 정도, 전체 20여 분."}</span>
              </div>
            </li>
            <li>
              <Circle size={18} />
              <div>
                <strong>가장 많이 틀린 검사부터 열어 보기</strong>
                <span>아래 목록이나 왼쪽 목록의 "일치도 낮은 순" 정렬로 열고, "차이 보기"에서 놓친 부위(파랑)와 더 그린 부위(빨강)를 확인합니다.</span>
              </div>
            </li>
          </ol>
        </section>
      )}

      <section className="neu-card overview-head" aria-label="모델 성능 요약">
        <div className="overview-score">
          <span className="score-label">
            학습에 쓰지 않은 검사 평균 일치도 <Help term="overview" />
          </span>
          {loading && !overview ? (
            <LoaderCircle className="spin" size={28} />
          ) : summary?.mean_dice != null ? (
            <>
              <strong className="score-value mono">{number(summary.mean_dice, 3)}</strong>
              <span className={`score-verdict v-${verdictClass(summary.mean_dice)}`}>{verdict(summary.mean_dice)}</span>
              <p className="muted">
                {overview?.predicted}개 검사, 종양 전체(영역 구분 없이) 기준 · 중앙값 {number(summary.median_dice, 2)}
                <br />
                영역별 평균의 평균 <b className="mono">{number(summary.mean_label_dice, 3)}</b>
                {model?.training?.validation_mean_dice != null && <> · 학습 중 측정값 <b className="mono">{number(model.training.validation_mean_dice, 3)}</b> (같은 기준)</>}
              </p>
            </>
          ) : (
            <>
              <strong className="score-value mono">—</strong>
              <p className="muted">아직 예측이 없습니다. 전체 예측을 실행하면 여기에 평균과 분포가 나타납니다.</p>
            </>
          )}
        </div>
        <div className="overview-actions">
          <button className={`neu-btn primary big ${running ? "busy" : ""}`} onClick={onBatch} disabled={!connected || running || (remaining === 0 && !running)}
            title={!connected ? "모델을 먼저 연결하세요" : remaining === 0 ? "모든 검증 검사에 예측이 있습니다" : undefined}>
            {running ? <LoaderCircle className="spin" size={18} /> : <PlayCircle size={18} />}
            {running ? `예측 중 ${batch!.done} / ${batch!.total}` : remaining > 0 ? `남은 ${remaining}개 검사 전체 예측` : "전체 예측 완료"}
          </button>
          {batch && (
            <div className="job-progress batch" aria-live="polite">
              <progress max={batch.total} value={batch.done} aria-label={`전체 예측 ${batch.done} / ${batch.total}`} />
              <small>
                {running ? `검사당 약 10초 · 남은 시간 약 ${Math.ceil(((batch.total - batch.done) * 10) / 60)}분` : `끝났습니다. ${batch.done - batch.failed}개 성공`}
                {batch.failed > 0 && ` · ${batch.failed}개 실패`}
              </small>
            </div>
          )}
          <span className="muted small-print">
            검증 검사 {overview?.candidates ?? candidates.length}개 중 {predicted}개 예측됨 · 모델 {model?.name ?? "미연결"}
          </span>
        </div>
      </section>

      {summary && scored.length > 0 && (
        <div className="overview-grid">
          <section className="neu-card" aria-label="일치도 분포">
            <header className="card-head">
              <div>
                <h3>일치도 분포 <Help term="dice" /></h3>
                <p>검사 하나가 막대 하나에 들어갑니다. 왼쪽에 쌓일수록 모델이 어려워한 검사가 많다는 뜻입니다.</p>
              </div>
            </header>
            <div className="histogram" role="img" aria-label={summary.histogram.map((b) => `${b.from.toFixed(1)}–${b.to.toFixed(1)}: ${b.count}개`).join(", ")}>
              {summary.histogram.map((bucket, i) => (
                <div key={i} className="hist-col">
                  <span className="hist-count mono">{bucket.count || ""}</span>
                  <span className={`hist-bar v-${verdictClass((bucket.from + bucket.to) / 2)}`} style={{ height: `${Math.max(2, (bucket.count / maxCount) * 100)}%` }} />
                  <span className="hist-label mono">{bucket.from.toFixed(1)}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="neu-card" aria-label="영역별 일치도">
            <header className="card-head">
              <div>
                <h3>영역별 평균 일치도</h3>
                <p>판독 마스크에 그 영역이 있는 검사만 셉니다. 작은 영역일수록 점수가 낮게 나오는 편입니다.</p>
              </div>
            </header>
            <ul className="label-bars">
              {labels.map((label) => (
                <li key={label.id}>
                  <span className="label-name"><span className="swatch" style={{ background: label.color }} /> {label.friendly}</span>
                  <span className="label-track">
                    <span className={`label-fill v-${verdictClass(label.stat?.mean)}`} style={{ width: `${((label.stat?.mean ?? 0) * 100).toFixed(1)}%` }} />
                  </span>
                  <span className="label-value mono">{label.stat?.mean != null ? number(label.stat.mean, 2) : "—"}</span>
                  <small className="muted">{label.stat?.count ?? 0}개</small>
                </li>
              ))}
            </ul>
          </section>

          <section className="neu-card worst" aria-label="일치도가 낮은 검사">
            <header className="card-head">
              <div>
                <h3>먼저 볼 검사 (일치도 낮은 순)</h3>
                <p>모델이 가장 많이 틀린 검사입니다. 눌러서 어디를 놓쳤는지 확인하세요.</p>
              </div>
            </header>
            <ol className="worst-list">
              {worst.map((row) => (
                <li key={row.case_id}>
                  <button className="worst-row" onClick={() => onOpenCase(row.case_id)}>
                    <span className="worst-name">{visitName(row.case_id, row.name)}</span>
                    <span className={`dice-mark v-${verdictClass(row.dice)}`}><span className="dot" /> {number(row.dice, 2)}</span>
                    <span className="worst-detail muted">
                      종양 {number(row.reference_volume_ml, 0)} mL · 놓침 {number(row.missed_ml, 1)} · 더 그림 {number(row.extra_ml, 1)}
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          </section>
        </div>
      )}
    </div>
  );
}
