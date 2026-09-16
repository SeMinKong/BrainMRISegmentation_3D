import { useEffect, useRef, useState } from "react";
import { BookOpen, CircleHelp } from "lucide-react";
import { GLOSSARY } from "../labels";

/** A small "?" next to a term. Click opens the plain-language explanation; Escape or an outside click closes it. */
export function Help({ term }: { term: string }) {
  const entry = GLOSSARY[term];
  const [open, setOpen] = useState(false);
  const wrapper = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent | KeyboardEvent) => {
      if (event instanceof KeyboardEvent ? event.key === "Escape" : !wrapper.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);
  if (!entry) return null;
  return (
    <span className="help" ref={wrapper}>
      <button type="button" className="help-btn" aria-label={`${entry.title} 설명`} aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        <CircleHelp size={14} />
      </button>
      {open && (
        <span className="help-pop" role="note">
          <strong>{entry.title}</strong>
          <span>{entry.text}</span>
        </span>
      )}
    </span>
  );
}

/** Collapsible list of every term, for reading through once. */
export function Glossary() {
  return (
    <details className="neu-card glossary">
      <summary>
        <BookOpen size={16} /> 용어 설명 한눈에 보기
      </summary>
      <dl>
        {Object.entries(GLOSSARY).map(([id, entry]) => (
          <div key={id}>
            <dt>{entry.title}</dt>
            <dd>{entry.text}</dd>
          </div>
        ))}
      </dl>
    </details>
  );
}
