import { ArrowDownWideNarrow, Plus, Search } from "lucide-react";
import { number } from "../api";
import type { Case } from "../api";
import { caseDice, caseVolume, findPatient, hasPrediction, patientDice, patientVolume } from "../patients";
import type { Patient, PatientFilter } from "../patients";
import { Help } from "./Help";
import { verdictClass } from "./ResultCard";

type Props = {
  patients: Patient[];
  /** Already filtered and sorted by the parent, which also uses the order for previous/next. */
  visible: Patient[];
  filter: PatientFilter;
  onFilter: (patch: Partial<PatientFilter>) => void;
  caseId: string;
  onSelect: (caseId: string) => void;
  onImport: () => void;
};

const SPLITS: { id: PatientFilter["split"]; name: string }[] = [
  { id: "val", name: "학습에 안 쓴 검사" },
  { id: "train", name: "학습에 쓴 검사" },
  { id: "all", name: "전체" },
];
const SORTS: { id: PatientFilter["sort"]; name: string; hint: string }[] = [
  { id: "id", name: "번호순", hint: "환자 번호 순서" },
  { id: "dice", name: "일치도 낮은 순", hint: "모델이 가장 많이 틀린 검사부터" },
  { id: "volume", name: "종양 큰 순", hint: "판독 마스크 부피가 큰 검사부터" },
];

/** A coloured dot for a visit's Dice, with the number alongside so colour is never the only cue. */
export function DiceMark({ dice }: { dice: number | null }) {
  if (dice == null) return null;
  return (
    <span className={`dice-mark v-${verdictClass(dice)}`} title={`일치도 ${number(dice, 2)}`}>
      <span className="dot" /> {number(dice, 2)}
    </span>
  );
}

export default function PatientBrowser({ patients, visible, filter, onFilter, caseId, onSelect, onImport }: Props) {
  const hasSplits = patients.some((p) => p.split);
  const current = findPatient(patients, caseId);
  const shownVisits = visible.reduce((n, p) => n + p.timepoints.length, 0);
  const sorted = filter.sort !== "id";

  return (
    <div className="patient-browser">
      <div className="browser-head">
        <h2>환자</h2>
        <span className="count-tag">{visible.length}명 · {shownVisits}회</span>
        <button className="neu-icon" aria-label="새 MRI 가져오기" title="MRI 가져오기" onClick={onImport}>
          <Plus size={16} />
        </button>
      </div>
      <label className="search-field">
        <Search size={15} />
        <input
          type="search"
          placeholder="환자 번호 검색"
          aria-label="환자 검색"
          value={filter.query}
          onChange={(e) => onFilter({ query: e.target.value })}
        />
      </label>
      {hasSplits && (
        <div className="chip-row" role="group" aria-label="목록 필터">
          {SPLITS.map((s) => (
            <button key={s.id} className={`chip ${filter.split === s.id ? "pressed" : ""}`} aria-pressed={filter.split === s.id} onClick={() => onFilter({ split: s.id })}>
              {s.name}
            </button>
          ))}
          <button className={`chip ${filter.predictedOnly ? "pressed" : ""}`} aria-pressed={filter.predictedOnly} onClick={() => onFilter({ predictedOnly: !filter.predictedOnly })}>
            예측 완료만
          </button>
          <button className={`chip ${filter.withCavity ? "pressed" : ""}`} aria-pressed={filter.withCavity} onClick={() => onFilter({ withCavity: !filter.withCavity })} title="수술 후 절제강이 있는 검사만">
            절제강 있음
          </button>
        </div>
      )}
      <label className="sort-field">
        <ArrowDownWideNarrow size={15} />
        <select className="neu-input" aria-label="정렬" value={filter.sort} onChange={(e) => onFilter({ sort: e.target.value as PatientFilter["sort"] })}>
          {SORTS.map((s) => (
            <option key={s.id} value={s.id} title={s.hint}>{s.name}</option>
          ))}
        </select>
      </label>
      {hasSplits && filter.split === "val" && !sorted && (
        <p className="browser-hint">
          모델이 학습 때 보지 못한 환자들입니다. 여기서의 결과가 실제 성능에 가깝습니다. <Help term="unseen" />
        </p>
      )}
      {filter.sort === "dice" && <p className="browser-hint">예측이 끝난 검사만 점수가 있습니다. 점수가 낮은 검사가 위에 옵니다.</p>}
      <ul className="patient-list" aria-label="환자 목록">
        {visible.length === 0 && <li className="muted">조건에 맞는 환자가 없습니다.</li>}
        {visible.map((patient) => {
          const active = patient.id === current?.id;
          const dice = patientDice(patient);
          const volume = patientVolume(patient);
          return (
            <li key={patient.id} className={`patient-row ${active ? "active" : ""}`}>
              <button className="patient-name" onClick={() => onSelect(patient.timepoints[0].case.id)} aria-current={active ? "true" : undefined}>
                <span className="patient-line">
                  <strong>{patient.name}</strong>
                  <DiceMark dice={dice} />
                </span>
                <small>
                  {patient.timepoints.length}회 검사
                  {volume != null && <> · 종양 최대 {number(volume, 0)} mL</>}
                </small>
              </button>
              <div className="timepoint-row" role="group" aria-label={`${patient.name} 검사 시점`}>
                {patient.timepoints.map((t) => {
                  const visitDice = caseDice(t.case);
                  const visitVolume = caseVolume(t.case);
                  const details = [visitDice != null ? `일치도 ${number(visitDice, 2)}` : hasPrediction(t.case) ? "예측 완료" : "", visitVolume != null ? `${number(visitVolume, 0)} mL` : ""].filter(Boolean).join(" · ");
                  return (
                    <button
                      key={t.case.id}
                      className={`timepoint ${t.case.id === caseId ? "pressed" : ""} ${hasPrediction(t.case) ? "predicted" : ""} ${visitDice != null ? `v-${verdictClass(visitDice)}` : ""}`}
                      aria-pressed={t.case.id === caseId}
                      title={details ? `${t.case.name} · ${details}` : t.case.name}
                      onClick={() => onSelect(t.case.id)}
                    >
                      {t.label}
                      {sorted && visitDice != null && <em className="mono">{number(visitDice, 2)}</em>}
                    </button>
                  );
                })}
              </div>
            </li>
          );
        })}
      </ul>
      <p className="browser-legend muted">
        <span className="dot" /> 예측 완료 · 일치도 <span className="v-good">0.7↑</span> <span className="v-mid">0.5↑</span> <span className="v-low">미만</span>
      </p>
    </div>
  );
}

export const describeCase = (data: Case) =>
  data.demo ? "합성 데모" : data.source === "linked" ? "MU-Glioma-Post 원본" : "가져온 MRI";
