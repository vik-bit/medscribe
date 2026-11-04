"""Render results JSON as Markdown tables (the README tables are produced by this)."""

from __future__ import annotations

from typing import Any

COLS = [
    ("note_f1", "Note token-F1", 3),
    ("highlight_recall", "Highlight recall", 3),
    ("ungrounded_lexical", "Ungrounded (lexical)", 3),
    ("unsupported_judge", "Unsupported (LLM judge)", 3),
    ("speaker_correct", "Speaker roles correct", 2),
    ("claims", "Claims/note", 1),
    ("latency_s", "Latency s", 0),
    ("prompt_tokens", "Prompt tok", 0),
    ("completion_tokens", "Output tok", 0),
    ("llm_calls", "LLM calls", 1),
]


def _fmt(m: dict[str, float] | None, digits: int) -> str:
    if not m:
        return "-"
    return f"{m['mean']:.{digits}f} [{m['lo']:.{digits}f}, {m['hi']:.{digits}f}]"


def notes_table(res: dict[str, Any]) -> str:
    names = list(res["metrics"])
    cols = [c for c in COLS if any(c[0] in res["metrics"][n] for n in names)]
    head = "| Pipeline | n | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|---|" + "---|" * len(cols)
    lines = [head, sep]
    for n in names:
        m = res["metrics"][n]
        lines.append(
            f"| {n} | {res['n'][n]} | " + " | ".join(_fmt(m.get(k), d) for k, _, d in cols) + " |"
        )
    lines.append("")
    lines.append("Paired difference vs `single` (mean, 95% bootstrap CI over consultations):")
    lines.append("")
    lines.append("| Pipeline | Metric | Diff | 95% CI |")
    lines.append("|---|---|---|---|")
    for n, metrics in res.get("paired", {}).items():
        for k, v in metrics.items():
            lines.append(f"| {n} | {k} | {v['diff']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] |")
    return "\n".join(lines)
