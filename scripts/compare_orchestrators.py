"""Hand-written vs LangGraph orchestration: equivalence, overhead, cost, code size.

    python scripts/compare_orchestrators.py loc
    python scripts/compare_orchestrators.py overhead --n 200
    python scripts/compare_orchestrators.py llm --split test   # real LLM (Ollama), slow
    python scripts/compare_orchestrators.py report             # combine into results/

* ``loc``: logical lines (non-blank, non-comment, docstrings excluded) of the
  orchestration code only; the shared specialists and LLM layer are not counted.
* ``overhead``: an instant scripted LLM, so only orchestration time is measured.
* ``llm``: both pipelines on real consultations. The order alternates per
  consultation and Ollama's prompt cache is flushed before every run, so neither
  version inherits the other's KV cache. Per-run orchestration overhead is
  wall time minus time spent inside LLM calls, which cancels LLM speed and any
  queueing behind other users of the same Ollama server.
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics as st
import time
from pathlib import Path

from medscribe.agents.llm import ScriptedLLM, make_llm
from medscribe.agents.messages import SoapNote
from medscribe.agents.orchestrator import run_multi
from medscribe.agents.runtime import Budget
from medscribe.data.primock import consultation_ids, load_consultation, make_splits
from medscribe.eval.note_metrics import highlight_recall, token_f1, ungrounded_rate
from medscribe.eval.run_notes import to_transcript
from medscribe.stats import bootstrap_ci, paired_bootstrap_diff

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "medscribe" / "agents"
OUT = ROOT / "results" / "orchestrators"
REVIEW_ONLY = {"_human_review", "ReviewSession", "start_review"}


# --- lines of code ------------------------------------------------------------
def _logical_lines(path: Path, names: set[str] | None = None, exclude: set[str] = frozenset()):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):  # fmt: skip
            doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    keep: set[int] = set()
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue  # imports are not orchestration logic
        if names is not None and name not in names:
            continue
        if name in exclude:
            continue
        keep.update(range(node.lineno, node.end_lineno + 1))
    lines = src.splitlines()
    return sum(1 for i in keep - doc_lines
               if lines[i - 1].strip() and not lines[i - 1].strip().startswith("#"))  # fmt: skip


def cmd_loc(_args) -> dict:
    res = {
        "hand_written_run_multi": _logical_lines(SRC / "orchestrator.py", {"run_multi"}),
        "langgraph_equivalent": _logical_lines(
            SRC / "langgraph_orchestrator.py", None, REVIEW_ONLY
        ),
        "langgraph_human_review_extra": _logical_lines(
            SRC / "langgraph_orchestrator.py", REVIEW_ONLY
        ),
        "note": "logical lines, imports and docstrings excluded; shared specialists not counted",
    }
    print(json.dumps(res, indent=1))
    return res


# --- orchestration overhead with an instant LLM ---------------------------------
def _realistic_responder(stored: dict, gold: dict):
    note = json.dumps(stored["note"])

    def respond(system: str, user: str) -> str:
        if "Plan the documentation work" in user:
            return json.dumps(
                {"steps": ["speaker", "extractor", "writer"], "presenting_complaint": "x"}
            )
        if "Decide which one is the DOCTOR" in user:
            return json.dumps(gold)
        if "List the clinically relevant facts" in user:
            return json.dumps(
                {"entities": [{"type": "symptom", "text": "pain", "evidence": [1]}] * 15}
            )
        if "A verifier rejected" in user:
            return json.dumps({"revised": []})
        return note

    return respond


def cmd_overhead(args) -> dict:
    from langgraph.checkpoint.memory import InMemorySaver

    from medscribe.agents.langgraph_orchestrator import build_graph, run_multi_langgraph

    root = Path(args.data)
    cid = make_splits(consultation_ids(root))["test"][0]
    t, gold = to_transcript(load_consultation(root, cid))
    stored = json.loads((ROOT / "results/runs/notes_test" / f"{cid}.multi.json").read_text())
    resp = _realistic_responder(stored, gold)
    plain = build_graph()
    saver = InMemorySaver()
    ckpt = build_graph(checkpointer=saver)

    t0 = time.perf_counter()
    for _ in range(20):
        build_graph()
    build_ms = (time.perf_counter() - t0) / 20 * 1000

    variants = {
        "hand_written": lambda: run_multi(ScriptedLLM(resp), t, budget=Budget()),
        "langgraph": lambda: run_multi_langgraph(ScriptedLLM(resp), t, graph=plain),
        "langgraph+InMemorySaver": lambda: run_multi_langgraph(
            ScriptedLLM(resp), t, graph=ckpt, checkpointer=saver),
    }  # fmt: skip
    times: dict[str, list[float]] = {k: [] for k in variants}
    for _ in range(5):  # warm-up
        for f in variants.values():
            f()
    for i in range(args.n):  # interleave variants to share any machine noise
        for k in list(variants) if i % 2 == 0 else list(reversed(variants)):
            t1 = time.perf_counter()
            variants[k]()
            times[k].append((time.perf_counter() - t1) * 1000)
    res = {"consultation": cid, "turns": len(t.turns), "n": args.n, "graph_build_ms": build_ms}
    for k, v in times.items():
        v.sort()
        res[k] = {"median_ms": st.median(v), "p95_ms": v[int(0.95 * len(v)) - 1]}
    print(json.dumps(res, indent=1))
    return res


# --- real LLM -------------------------------------------------------------------
def _flush_cache(llm) -> None:
    """Replace the server's cached prompt prefix with something unrelated."""
    try:
        llm.complete("You are a calculator.", "2+2=", schema=None, timeout_s=120)
    except Exception:  # noqa: BLE001 - best effort
        pass


def cmd_llm(args) -> None:
    from medscribe.agents.langgraph_orchestrator import run_multi_langgraph

    root = Path(args.data)
    ids = make_splits(consultation_ids(root))[args.split][: args.limit or None]
    llm = make_llm(args.llm)
    runs = OUT / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    fns = {"multi": run_multi, "multi-langgraph": run_multi_langgraph}
    for i, cid in enumerate(ids):
        t, gold = to_transcript(load_consultation(root, cid))
        order = list(fns) if i % 2 == 0 else list(reversed(fns))
        for name in order:
            path = runs / f"{cid}.{name}.json"
            if path.exists():
                continue
            _flush_cache(llm)
            r = fns[name](llm, t, budget=Budget())
            s = r.summary()
            llm_s = sum(a["latency_s"] for a in s["per_agent"].values())
            rec = {"cid": cid, "llm": llm.name, "order": order.index(name), "summary": s,
                   "llm_time_s": llm_s, "overhead_s": s["latency_s"] - llm_s,
                   "gold_speaker_map": gold, "note": r.note.model_dump(),
                   "trace": r.trace.to_json()}  # fmt: skip
            path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            print(f"{cid} {name}: {r.status} claims={len(r.note.claims())} {s['latency_s']:.0f}s "
                  f"(overhead {rec['overhead_s'] * 1000:.0f} ms)", flush=True)  # fmt: skip


def _ci(vals):
    m, lo, hi = bootstrap_ci(vals)
    return {"mean": m, "lo": lo, "hi": hi, "n": len(vals)}


def cmd_report(args) -> dict:
    root = Path(args.data)
    runs = OUT / "runs"
    stored_dir = ROOT / "results/runs/notes_test"
    rows: dict[str, list[dict]] = {"multi": [], "multi-langgraph": []}
    same_note = same_calls = same_as_stored = n_pairs = 0
    for cid in make_splits(consultation_ids(root))[args.split]:
        recs = {}
        for name in rows:
            p = runs / f"{cid}.{name}.json"
            if p.exists():
                recs[name] = json.loads(p.read_text(encoding="utf-8"))
        if len(recs) < 2:
            continue
        c = load_consultation(root, cid)
        t, _ = to_transcript(c)
        turns = [x.text for x in t.turns]
        n_pairs += 1
        a, b = recs["multi"], recs["multi-langgraph"]
        same_note += a["note"] == b["note"]
        calls = lambda r: [(e["agent"], e["prompt_tokens"], e["completion_tokens"])  # noqa: E731
                           for e in r["trace"] if e["event"] in ("ok", "parse_error")]  # fmt: skip
        same_calls += calls(a) == calls(b)
        sp = stored_dir / f"{cid}.multi.json"
        if sp.exists():
            same_as_stored += json.loads(sp.read_text(encoding="utf-8"))["note"] == a["note"]
        for name, r in recs.items():
            note = SoapNote.model_validate(r["note"])
            text = note.render()
            rows[name].append({
                "cid": cid,
                "note_f1": token_f1(text, c.note)[2],
                "highlight_recall": highlight_recall(text, c.highlights),
                "ungrounded_lexical": ungrounded_rate([x.text for _, x in note.claims()], turns),
                "latency_s": r["summary"]["latency_s"],
                "llm_time_s": r["llm_time_s"],
                "overhead_ms": r["overhead_s"] * 1000,
                "prompt_tokens": r["summary"]["prompt_tokens"],
                "completion_tokens": r["summary"]["completion_tokens"],
                "llm_calls": r["summary"]["llm_calls"],
                "status": r["summary"]["status"],
            })  # fmt: skip
    res: dict = {"n_consultations": n_pairs, "identical_notes": same_note,
                 "identical_llm_call_sequence": same_calls,
                 "hand_written_identical_to_stored_nov4_run": same_as_stored, "metrics": {},
                 "paired_langgraph_minus_hand_written": {}}  # fmt: skip
    keys = ["note_f1", "highlight_recall", "ungrounded_lexical", "latency_s", "llm_time_s",
            "overhead_ms", "prompt_tokens", "completion_tokens", "llm_calls"]  # fmt: skip
    for name, rs in rows.items():
        res["metrics"][name] = {k: _ci([r[k] for r in rs if r[k] == r[k]]) for k in keys}
    by = {name: {r["cid"]: r for r in rs} for name, rs in rows.items()}
    for k in keys:
        pairs = [(by["multi-langgraph"][c][k], by["multi"][c][k]) for c in by["multi"]
                 if c in by["multi-langgraph"]]  # fmt: skip
        pairs = [(x, y) for x, y in pairs if x == x and y == y]
        if len(pairs) >= 2:
            d, lo, hi, _ = paired_bootstrap_diff([x for x, _ in pairs], [y for _, y in pairs])
            res["paired_langgraph_minus_hand_written"][k] = {"diff": d, "lo": lo, "hi": hi}
    for f in ("loc.json", "overhead.json"):
        if (OUT / f).exists():
            res[f[:-5]] = json.loads((OUT / f).read_text(encoding="utf-8"))
    res["rows"] = rows
    (OUT / "comparison.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["loc", "overhead", "llm", "report"])
    ap.add_argument("--data", default=str(ROOT / "data" / "primock57"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--llm", default="ollama:qwen2.5:3b")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"loc": cmd_loc, "overhead": cmd_overhead, "llm": cmd_llm, "report": cmd_report}[
        args.cmd
    ](args)
    if args.cmd in ("loc", "overhead"):
        (OUT / f"{args.cmd}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
