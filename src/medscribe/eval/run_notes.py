"""Generate notes with each pipeline, then score them.

Generation is resumable: each (consultation, pipeline) output is a JSON file
under ``runs/<tag>/``; existing files are skipped. ``single+verify`` is derived
from the *same* single-agent LLM output (verifier applied afterwards), so the
two are exactly paired and the ablation isolates the verifier.
"""

from __future__ import annotations

import json
import math
import time
import zlib
from pathlib import Path
from typing import Any

from medscribe.agents.llm import LLM, FaultInjectingLLM
from medscribe.agents.messages import SoapNote, Transcript, Turn
from medscribe.agents.orchestrator import run_multi, run_single
from medscribe.agents.runtime import Budget
from medscribe.agents.specialists import drop_rejected, verify
from medscribe.data.primock import Consultation
from medscribe.eval.judge import judge_claims
from medscribe.eval.note_metrics import (
    highlight_recall,
    split_reference_claims,
    token_f1,
    ungrounded_rate,
)
from medscribe.stats import bootstrap_ci, paired_bootstrap_diff


def to_transcript(c: Consultation, seed: int = 0) -> tuple[Transcript, dict[str, str]]:
    utts, gold = c.diarized(seed)
    return Transcript(
        cid=c.cid, turns=[Turn(uid=u.uid, speaker=u.speaker, text=u.text) for u in utts]
    ), gold


def _dump(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def generate(
    consultations: list[Consultation],
    llm: LLM,
    out: Path,
    pipelines: list[str],
    *,
    fault_rate: float = 0.0,
    max_tokens: int = 60_000,
    max_steps: int = 12,
    log=print,
) -> None:
    for c in consultations:
        t, gold = to_transcript(c)
        for name in pipelines:
            if name == "single+verify":
                continue  # derived from "single" below
            path = out / f"{c.cid}.{name}.json"
            if path.exists():
                continue
            model = (
                FaultInjectingLLM(llm, fault_rate, seed=zlib.crc32(c.cid.encode()))
                if fault_rate
                else llm
            )
            budget = Budget(max_tokens=max_tokens, max_steps=max_steps)
            t0 = time.perf_counter()
            if name == "single":
                r = run_single(model, t, budget=budget)
            elif name == "multi":
                r = run_multi(model, t, budget=budget)
            else:
                raise ValueError(name)
            rec = {
                "cid": c.cid,
                "llm": llm.name,
                "fault_rate": fault_rate,
                "injected": getattr(model, "injected", []),
                "gold_speaker_map": gold,
                "summary": r.summary(),
                "note": r.note.model_dump(),
                "trace": r.trace.to_json(),
            }
            _dump(path, rec)
            log(f"{c.cid} {name}: {r.status} claims={len(r.note.claims())} "
                f"{time.perf_counter() - t0:.0f}s")  # fmt: skip
        if "single+verify" in pipelines:
            src, dst = out / f"{c.cid}.single.json", out / f"{c.cid}.single+verify.json"
            if src.exists() and not dst.exists():
                rec = json.loads(src.read_text(encoding="utf-8"))
                note = SoapNote.model_validate(rec["note"])
                t0 = time.perf_counter()
                report = verify(t, note)
                kept = drop_rejected(note, report)
                s = dict(rec["summary"])
                s.update(pipeline="single+verify", claims=len(kept.claims()),
                         dropped=len(report.rejected),
                    recited=sum(v.recited for v in report.verdicts), rejected_per_round=[len(report.rejected)],
                         latency_s=round(s["latency_s"] + time.perf_counter() - t0, 2))  # fmt: skip
                _dump(dst, {**rec, "summary": s, "note": kept.model_dump()})


def score(
    consultations: list[Consultation],
    out: Path,
    pipelines: list[str],
    *,
    judge: LLM | None = None,
    log=print,
) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {p: [] for p in pipelines}
    for c in consultations:
        t, gold = to_transcript(c)
        turns = [x.text for x in t.turns]
        for name in pipelines:
            path = out / f"{c.cid}.{name}.json"
            if not path.exists():
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            note = SoapNote.model_validate(rec["note"])
            claims = [cl.text for _, cl in note.claims()]
            text = note.render()
            _, _, f1 = token_f1(text, c.note)
            row = {
                "cid": c.cid,
                "note_f1": f1,
                "highlight_recall": highlight_recall(text, c.highlights),
                "ungrounded_lexical": ungrounded_rate(claims, turns),
                **{k: rec["summary"][k] for k in ("claims", "latency_s", "prompt_tokens",
                                                   "completion_tokens", "llm_calls", "status")},
            }  # fmt: skip
            sm = rec["summary"].get("speaker_map")
            if sm:
                row["speaker_correct"] = float(sm == rec["gold_speaker_map"])
            if judge is not None and claims:
                if "judge" not in rec:
                    v, usage = judge_claims(judge, claims, turns)
                    rec["judge"] = {"verdicts": v, "usage": usage, "model": judge.name}
                    _dump(path, rec)
                v = [x for x in rec["judge"]["verdicts"] if x is not None]
                row["unsupported_judge"] = (sum(not x for x in v) / len(v)) if v else math.nan
                row["judged_fraction"] = len(v) / len(claims)
            rows[name].append(row)
        if judge is not None:  # clinician note as a calibration row for both grounding metrics
            ref_path = out / f"{c.cid}.reference_judge.json"
            ref_claims = split_reference_claims(c.note)
            if not ref_path.exists():
                v, usage = judge_claims(judge, ref_claims, turns)
                _dump(ref_path, {"claims": ref_claims, "verdicts": v, "usage": usage})
            ref = json.loads(ref_path.read_text(encoding="utf-8"))
            v = [x for x in ref["verdicts"] if x is not None]
            rows.setdefault("reference", []).append({
                "cid": c.cid,
                "ungrounded_lexical": ungrounded_rate(ref_claims, turns),
                "unsupported_judge": (sum(not x for x in v) / len(v)) if v else math.nan,
                "claims": len(ref_claims),
            })  # fmt: skip
        log(f"scored {c.cid}")
    return aggregate(rows)


METRICS = ["note_f1", "highlight_recall", "ungrounded_lexical", "unsupported_judge",
           "speaker_correct", "claims", "latency_s", "prompt_tokens", "completion_tokens",
           "llm_calls"]  # fmt: skip


def aggregate(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": {k: len(v) for k, v in rows.items()}, "metrics": {}, "paired": {}}
    for name, rs in rows.items():
        m = {}
        for key in METRICS:
            vals = [r[key] for r in rs if key in r and not _nan(r[key])]
            if vals:
                mean, lo, hi = bootstrap_ci(vals)
                m[key] = {"mean": mean, "lo": lo, "hi": hi, "n": len(vals)}
        m["status"] = _count([r.get("status", "") for r in rs])
        out["metrics"][name] = m
    base = {r["cid"]: r for r in rows.get("single", [])}
    for name, rs in rows.items():
        if name in ("single", "reference"):
            continue
        out["paired"][name] = {}
        for key in ["note_f1", "highlight_recall", "ungrounded_lexical", "unsupported_judge",
                    "latency_s", "prompt_tokens"]:  # fmt: skip
            pairs = [(r[key], base[r["cid"]][key]) for r in rs
                     if r["cid"] in base and key in r and key in base[r["cid"]]
                     and not _nan(r[key]) and not _nan(base[r["cid"]][key])]  # fmt: skip
            if len(pairs) >= 2:
                d, lo, hi, p_le0 = paired_bootstrap_diff(
                    [a for a, _ in pairs], [b for _, b in pairs]
                )
                out["paired"][name][key] = {"diff": d, "lo": lo, "hi": hi, "p_le_0": p_le0,
                                            "n": len(pairs)}  # fmt: skip
    out["rows"] = rows
    return out


def _nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _count(xs: list[str]) -> dict[str, int]:
    d: dict[str, int] = {}
    for x in xs:
        if x:
            d[x] = d.get(x, 0) + 1
    return d
