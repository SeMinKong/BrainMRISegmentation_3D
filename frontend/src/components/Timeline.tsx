import { useEffect, useState } from "react";
import { LoaderCircle } from "lucide-react";
import { api, number } from "../api";
import type { Case, Stats } from "../api";
import { friendlyLabel } from "../labels";
import type { Patient } from "../patients";
import { Help } from "./Help";

type Props = {
  patient: Patient;
  data: Case;
  visibleLabels: number[];
  onSelect: (caseId: string) => void;
};

/** Per-visit tumour volumes (expert mask) for one patient; click a bar to open that visit. */
export default function Timeline({ patient, data, visibleLabels, onSelect }: Props) {
  const [volumes, setVolumes] = useState<Record<string, Stats | null>>({});
  const ids = patient.timepoints.map((t) => t.case.id).join("|");
  useEffect(() => {
    const controller = new AbortController();
    setVolumes({});
    Promise.all(
      patient.timepoints.map((t) =>
        t.case.segmentations.some((s) => s.id === "reference")
          ? api<Stats>(`/cases/${t.case.id}/stats`, { signal: controller.signal }).catch(() => null)
          : Promise.resolve(null),
      ),
    ).then((results) => {
      if (controller.signal.aborted) return;
      setVolumes(Object.fromEntries(patient.timepoints.map((t, i) => [t.case.id, results[i]])));
    });
    return () => controller.abort();
  }, [ids]);

  const loaded = Object.keys(volumes).length > 0;
  const shown = (stats: Stats | null) =>
    stats ? stats.regions.filter((r) => visibleLabels.includes(r.label)).reduce((sum, r) => sum + r.volume_ml, 0) : null;
  const max = Math.max(0.001, ...patient.timepoints.map((t) => shown(volumes[t.case.id] ?? null) ?? 0));

  return (
    <section className="neu-card timeline" aria-label="검사 시점별 종양 부피">
      <header className="card-head">
        <div>
          <h3>이 환자의 검사 시점별 부피 <Help term="volume" /></h3>
          <p>판독 마스크 기준, 표시 중인 영역의 합계(mL). 막대를 누르면 그 검사로 이동합니다.</p>
        </div>
        {!loaded && <LoaderCircle className="spin" size={18} aria-label="부피 계산 중" />}
      </header>
      <div className="timeline-bars">
        {patient.timepoints.map((t) => {
          const stats = volumes[t.case.id] ?? null;
          const total = shown(stats);
          const current = t.case.id === data.id;
          return (
            <div key={t.case.id} className={`timeline-col ${current ? "current" : ""}`}>
              <button
                className="timeline-bar-btn"
                aria-label={`${t.label} 검사${total != null ? `, ${number(total, 1)} mL` : ""}${current ? ", 현재 보고 있는 검사" : ""}`}
                aria-current={current ? "true" : undefined}
                onClick={() => onSelect(t.case.id)}
              >
                <span className="timeline-value">{total == null ? (loaded ? "—" : "") : number(total, 1)}</span>
                <span className="timeline-track">
                  <span className="timeline-stack" style={{ height: `${total == null ? 0 : Math.max(3, (total / max) * 100)}%` }}>
                    {stats?.regions
                      .filter((r) => visibleLabels.includes(r.label) && r.volume_ml > 0)
                      .map((r) => (
                        <span
                          key={r.label}
                          style={{ background: r.color, flex: r.volume_ml }}
                          title={`${friendlyLabel({ id: r.label, name: r.name, color: r.color }, data)} ${number(r.volume_ml, 1)} mL`}
                        />
                      ))}
                  </span>
                </span>
                <span className="timeline-label">{t.label}</span>
              </button>
            </div>
          );
        })}
      </div>
    </section>
  );
}
