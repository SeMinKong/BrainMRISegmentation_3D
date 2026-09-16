import { useEffect, useRef, useState } from "react";
import { Check, Files, LoaderCircle, Upload, X } from "lucide-react";
import { api, number } from "../api";
import type { Case } from "../api";

type Props = { onClose: () => void; onImported: (data: Case) => void };

export default function ImportDialog({ onClose, onImported }: Props) {
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
    setError(accepted.length === items.length ? "" : ".nii 또는 .nii.gz 파일만 사용할 수 있습니다.");
  };
  const submit = async () => {
    setBusy(true);
    setError("");
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    form.append("label_preset", preset);
    if (name.trim()) form.append("name", name.trim());
    try {
      onImported(await api<Case>("/cases/import", { method: "POST", body: form }));
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
      <div className="dialog-head">
        <h2>MRI 가져오기</h2>
        <button className="neu-icon" onClick={onClose} disabled={busy} aria-label="가져오기 창 닫기">
          <X size={18} />
        </button>
      </div>
      <p className="muted">한 검사의 MRI 파일과, 있다면 종양 마스크를 함께 선택하세요.</p>
      <label className="field-label" htmlFor="case-name">
        이름 <span>선택 사항</span>
      </label>
      <input
        id="case-name"
        className="neu-input"
        placeholder="예: 환자001 2차 검사"
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
        <Files size={26} />
        <strong>파일을 놓거나 눌러서 선택</strong>
        <span>.nii / .nii.gz · 파일당 최대 256 MB</span>
        <input type="file" accept=".nii,.gz" multiple disabled={busy} onChange={(e) => accept(Array.from(e.target.files || []))} />
      </label>
      {files.length > 0 && (
        <ul className="file-list">
          {files.map((f) => (
            <li key={f.name}>
              <Check size={14} />
              <span>{f.name}</span>
              <small>{number(f.size / 1024 / 1024, 1)} MB</small>
            </li>
          ))}
        </ul>
      )}
      <label className="field-label" htmlFor="label-preset">
        마스크 라벨 체계
      </label>
      <select className="neu-input" id="label-preset" value={preset} disabled={busy} onChange={(e) => setPreset(e.target.value)}>
        <option value="mu_glioma_post">MU-Glioma-Post · 1 NETC / 2 SNFH / 3 ET / 4 RC</option>
        <option value="generic">사용자 데이터 · 라벨 1–4</option>
      </select>
      <p className="muted small-print">
        파일 이름 끝의 <code>t1n</code>, <code>t1c</code>, <code>t2w</code>, <code>t2f</code>, <code>tumorMask</code>(또는 <code>seg</code>)로 시퀀스를
        구분합니다. 라벨 번호는 자동 변환하지 않습니다. 영상은 이 컴퓨터에만 저장됩니다.
      </p>
      {error && (
        <div role="alert" className="inline-error">
          {error}
        </div>
      )}
      <div className="dialog-foot">
        <button className="neu-btn" onClick={onClose} disabled={busy}>
          취소
        </button>
        <button className="neu-btn primary" disabled={!files.length || busy} onClick={submit}>
          {busy ? <LoaderCircle className="spin" size={16} /> : <Upload size={16} />} {busy ? "검증 중…" : "추가"}
        </button>
      </div>
    </dialog>
  );
}
