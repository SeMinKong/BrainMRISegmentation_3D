import { useMemo, useState } from "react";
import { Plus, Search } from "lucide-react";
import type { Case } from "../api";
import { filterPatients, findPatient, hasPrediction } from "../patients";
import type { Patient, PatientFilter } from "../patients";
import { Help } from "./Help";

type Props = {
  patients: Patient[];
  caseId: string;
  onSelect: (caseId: string) => void;
  onImport: () => void;
};

const SPLITS: { id: PatientFilter["split"]; name: string }[] = [
  { id: "val", name: "학습에 안 쓴 검사" },
  { id: "train", name: "학습에 쓴 검사" },
  { id: "all", name: "전체" },
];

export default function PatientBrowser({ patients, caseId, onSelect, onImport }: Props) {
  const hasSplits = patients.some((p) => p.split);
  const [filter, setFilter] = useState<PatientFilter>({ query: "", split: hasSplits ? "val" : "all", predictedOnly: false });
  const visible = useMemo(() => filterPatients(patients, filter), [patients, filter]);
  const current = findPatient(patients, caseId);
  const shownVisits = visible.reduce((n, p) => n + p.timepoints.length, 0);

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
          onChange={(e) => setFilter((f) => ({ ...f, query: e.target.value }))}
        />
      </label>
      {hasSplits && (
        <div className="chip-row" role="group" aria-label="목록 필터">
          {SPLITS.map((s) => (
            <button
              key={s.id}
              className={`chip ${filter.split === s.id ? "pressed" : ""}`}
              aria-pressed={filter.split === s.id}
              onClick={() => setFilter((f) => ({ ...f, split: s.id }))}
            >
              {s.name}
            </button>
          ))}
          <button
            className={`chip ${filter.predictedOnly ? "pressed" : ""}`}
            aria-pressed={filter.predictedOnly}
            onClick={() => setFilter((f) => ({ ...f, predictedOnly: !f.predictedOnly }))}
          >
            예측 완료만
          </button>
        </div>
      )}
      {hasSplits && filter.split === "val" && (
        <p className="browser-hint">
          모델이 학습 때 보지 못한 환자들입니다. 여기서의 결과가 실제 성능에 가깝습니다. <Help term="unseen" />
        </p>
      )}
      <ul className="patient-list" aria-label="환자 목록">
        {visible.length === 0 && <li className="muted">조건에 맞는 환자가 없습니다.</li>}
        {visible.map((patient) => {
          const active = patient.id === current?.id;
          return (
            <li key={patient.id} className={`patient-row ${active ? "active" : ""}`}>
              <button
                className="patient-name"
                onClick={() => onSelect(patient.timepoints[0].case.id)}
                aria-current={active ? "true" : undefined}
              >
                <strong>{patient.name}</strong>
                <small>{patient.timepoints.length}회 검사</small>
              </button>
              <div className="timepoint-row" role="group" aria-label={`${patient.name} 검사 시점`}>
                {patient.timepoints.map((t) => (
                  <button
                    key={t.case.id}
                    className={`timepoint ${t.case.id === caseId ? "pressed" : ""} ${hasPrediction(t.case) ? "predicted" : ""}`}
                    aria-pressed={t.case.id === caseId}
                    title={hasPrediction(t.case) ? `${t.case.name} · 예측 완료` : t.case.name}
                    onClick={() => onSelect(t.case.id)}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </li>
          );
        })}
      </ul>
      <p className="browser-legend muted">
        <span className="dot" /> 예측 완료
      </p>
    </div>
  );
}

export const describeCase = (data: Case) =>
  data.demo ? "합성 데모" : data.source === "linked" ? "MU-Glioma-Post 원본" : "가져온 MRI";
