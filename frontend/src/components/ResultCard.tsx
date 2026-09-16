import { ArrowDownToLine, Eye, EyeOff } from "lucide-react";
import { number } from "../api";
import type { Case, Overview, Segmentation, Stats } from "../api";
import { DIFF, friendlyLabel, labelTerm } from "../labels";
import { Help } from "./Help";

type Props = {
  data: Case;
  prediction: Segmentation | null;
  referenceStats: Stats | null;
  predictionStats: Stats | null;
  visibleLabels: number[];
  onToggleLabel: (id: number) => void;
  /** Validation-set results for the same model, so one score can be placed in the distribution. */
  overview?: Overview | null;
  /** Briefly highlighted after a prediction on this case finishes. */
  highlight?: boolean;
  onShowDiff?: () => void;
};

export const verdict = (dice: number | null | undefined) => {
  if (dice == null) return "";
  if (dice >= 0.9) return "매우 일치";
  if (dice >= 0.7) return "대체로 일치";
  if (dice >= 0.5) return "부분 일치";
  return "많이 다름";
};
export const verdictClass = (dice: number | null | undefined) => (dice != null && dice >= 0.7 ? "good" : dice != null && dice >= 0.5 ? "mid" : "low");

/** Where a score sits among the validation cases: share of cases that scored lower. */
export function percentile(dice: number, overview: Overview | null | undefined): { top: number; count: number } | null {
  const scores = (overview?.cases ?? []).map((c) => c.dice).filter((d): d is number => d != null);
  if (scores.length < 5) return null;
  const below = scores.filter((d) => d < dice).length;
  return { top: Math.max(1, Math.round((1 - below / scores.length) * 100)), count: scores.length };
}

/** What the reader came for: how close the model's mask is to the expert's, per region, in plain words. */
export default function ResultCard({ data, prediction, referenceStats, predictionStats, visibleLabels, onToggleLabel, overview, highlight, onShowDiff }: Props) {
  const hasReference = data.segmentations.some((s) => s.id === "reference");
  const dice = prediction && hasReference ? predictionStats?.dice : null;
  const download = (id: string) => `/api/cases/${data.id}/segmentations/${encodeURIComponent(id)}/download`;
  const rank = dice != null ? percentile(dice, overview) : null;
  const mean = overview?.summary.mean_dice;
  const histogram = overview?.summary.histogram ?? [];
  const bucketOf = dice != null ? histogram.findIndex((b) => dice >= b.from && dice < b.to) : -1;
  const maxCount = Math.max(1, ...histogram.map((b) => b.count));

  return (
    <section className={`neu-card result-card ${highlight ? "flash" : ""}`} aria-label="종양 영역 결과" id="result-card">
      <div className="result-head">
        {prediction && hasReference ? (
          <div className="score">
            <span className="score-label">
              전체 일치도 <Help term="dice" />
            </span>
            <strong className="score-value mono">{number(dice, 2)}</strong>
            <span className={`score-verdict v-${verdictClass(dice)}`}>{verdict(dice)}</span>
            <p className="muted">
              내 모델의 예측 <Help term="prediction" />과 판독 마스크 <Help term="reference" />가 겹치는 정도입니다.
              {predictionStats?.missed_ml != null && predictionStats?.extra_ml != null && (
                <>
                  {" "}<span className="diff-word" style={{ color: DIFF.missed.color }}>놓친 부위 {number(predictionStats.missed_ml, 1)} mL</span>
                  {" · "}<span className="diff-word" style={{ color: DIFF.extra.color }}>더 그린 부위 {number(predictionStats.extra_ml, 1)} mL</span>
                  <Help term="missed" />
                  {onShowDiff && <button className="link-btn" onClick={onShowDiff}>차이 보기로 확인</button>}
                </>
              )}
            </p>
          </div>
        ) : prediction ? (
          <div className="score">
            <span className="score-label">모델 예측만 있는 검사</span>
            <p className="muted">판독 마스크가 없어 일치도는 계산할 수 없고, 예측 부피만 표시합니다.</p>
          </div>
        ) : (
          <div className="score">
            <span className="score-label">
              판독 마스크 <Help term="reference" />
            </span>
            <p className="muted">아직 이 검사에는 모델 예측이 없습니다. 위의 버튼으로 예측하면 여기에 일치도가 표시됩니다.</p>
          </div>
        )}
        {dice != null && mean != null && histogram.length > 0 && (
          <div className="score-context" aria-label="검증 검사 전체와 비교">
            <span className="score-label">
              검증 검사 {rank?.count ?? overview?.predicted}개 중 <Help term="percentile" />
            </span>
            <div className="context-line">
              <span>이 검사 <b className="mono">{number(dice, 2)}</b></span>
              <span>검증 평균 <b className="mono">{number(mean, 2)}</b></span>
              {rank && (rank.top <= 50 ? <span>상위 <b className="mono">{rank.top}%</b></span> : <span>하위 <b className="mono">{101 - rank.top}%</b></span>)}
            </div>
            <div className="mini-histogram" aria-hidden="true">
              {histogram.map((bucket, i) => (
                <span key={i} className={`bar ${i === bucketOf ? "here" : ""}`} style={{ height: `${Math.max(8, (bucket.count / maxCount) * 100)}%` }} title={`${bucket.from.toFixed(1)}–${bucket.to.toFixed(1)}: ${bucket.count}개`} />
              ))}
            </div>
            <div className="histogram-axis mono" aria-hidden="true"><span>0</span><span>0.5</span><span>1</span></div>
          </div>
        )}
      </div>
      <table className="data-table result-table">
        <thead>
          <tr>
            <th>영역</th>
            <th>
              판독 <span className="unit">mL</span> <Help term="volume" />
            </th>
            {prediction && <th>예측 <span className="unit">mL</span></th>}
            {prediction && hasReference && <th>일치도</th>}
            <th className="th-eye">표시</th>
          </tr>
        </thead>
        <tbody>
          {data.labels.map((label) => {
            const ref = referenceStats?.regions.find((r) => r.label === label.id);
            const pred = predictionStats?.regions.find((r) => r.label === label.id);
            const on = visibleLabels.includes(label.id);
            const regionDice = pred?.dice;
            const refEmpty = (ref?.voxels ?? 0) === 0;
            const predEmpty = (pred?.voxels ?? 0) === 0;
            const absent = refEmpty && predEmpty;
            // One-sided cases read better as words than as a Dice of 0.
            const oneSided = refEmpty && !predEmpty ? "판독에는 없는 영역을 예측함" : !refEmpty && predEmpty ? "모델이 놓침" : "";
            const name = friendlyLabel(label, data);
            return (
              <tr key={label.id} className={on ? "" : "dimmed"}>
                <td>
                  <span className="swatch" style={{ background: label.color }} /> {name}
                  {labelTerm[label.id] && <Help term={labelTerm[label.id]} />}
                </td>
                <td className="mono">{hasReference ? number(ref?.volume_ml, 1) : "—"}</td>
                {prediction && <td className="mono">{number(pred?.volume_ml, 1)}</td>}
                {prediction && hasReference && (
                  <td className="mono">
                    {absent ? <span className="muted">해당 없음</span>
                      : oneSided ? <span className="one-sided">{oneSided}</span>
                        : regionDice == null ? "—" : `${number(regionDice, 2)}`}
                    {!absent && !oneSided && regionDice != null && <small className="muted"> {verdict(regionDice)}</small>}
                  </td>
                )}
                <td>
                  <button className="neu-icon small" aria-label={`${name} ${on ? "숨기기" : "표시"}`} aria-pressed={on} onClick={() => onToggleLabel(label.id)}>
                    {on ? <Eye size={15} /> : <EyeOff size={15} />}
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="result-foot">
        <span className="muted">
          {prediction && hasReference
            ? "\"해당 없음\"은 판독과 예측 모두에 그 영역이 없다는 뜻입니다. 절제강은 수술하지 않은 환자에게는 없습니다. 작은 영역은 몇 mL만 어긋나도 일치도가 크게 떨어집니다."
            : "영역별 부피는 색 표시된 점의 개수에 점 하나의 크기를 곱한 값입니다."}
        </span>
        <span className="result-links">
          {hasReference && <a href={download("reference")}><ArrowDownToLine size={13} /> 판독 마스크 저장</a>}
          {prediction && <a href={download(prediction.id)}><ArrowDownToLine size={13} /> 예측 마스크 저장</a>}
        </span>
      </div>
    </section>
  );
}
