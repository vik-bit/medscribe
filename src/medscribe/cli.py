"""Command line entry point: ``python -m medscribe <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_DATA = Path("data/primock57")


def _consultations(args):
    from medscribe.data.primock import consultation_ids, load_consultation, make_splits

    root = Path(args.data)
    ids = consultation_ids(root)
    if args.split != "all":
        ids = make_splits(ids)[args.split]
    if args.ids:
        ids = [i for i in ids if i in set(args.ids.split(","))]
    if args.limit:
        ids = ids[: args.limit]
    return [load_consultation(root, i) for i in ids]


def cmd_splits(args) -> None:
    from medscribe.data.primock import consultation_ids, make_splits

    s = make_splits(consultation_ids(Path(args.data)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(s, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in s.items()})


def cmd_notes(args) -> None:
    from medscribe.agents.llm import make_llm
    from medscribe.eval.report import notes_table
    from medscribe.eval.run_notes import generate, score

    cons = _consultations(args)
    pipelines = args.pipelines.split(",")
    out = Path(args.runs)
    llm = make_llm(args.llm)
    if not args.score_only:
        generate(cons, llm, out, pipelines, fault_rate=args.fault_rate,
                 max_tokens=args.max_tokens, max_steps=args.max_steps)  # fmt: skip
    judge = make_llm(args.judge) if args.judge else None
    res = score(cons, out, pipelines, judge=judge)
    res["config"] = {k: v for k, v in vars(args).items() if k != "func"}
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(notes_table(res))


def cmd_asr_eval(args) -> None:
    from medscribe.asr.evaluate import main as asr_eval

    asr_eval(args)


def cmd_asr_compare(args) -> None:
    from medscribe.asr.evaluate import compare

    compare(args)


def cmd_asr_train(args) -> None:
    from medscribe.asr.train import main as asr_train

    asr_train(args)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="medscribe")
    sub = ap.add_subparsers(required=True)

    def data_args(p):
        p.add_argument("--data", default=str(DEFAULT_DATA))
        p.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
        p.add_argument("--ids", default="", help="comma-separated consultation ids")
        p.add_argument("--limit", type=int, default=0)

    p = sub.add_parser("splits", help="write the consultation-level split")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--out", default="results/splits.json")
    p.set_defaults(func=cmd_splits)

    p = sub.add_parser("notes", help="generate + score notes for the pipelines")
    data_args(p)
    p.add_argument("--llm", default="ollama:qwen2.5:3b")
    p.add_argument("--judge", default="", help="LLM spec for the claim-support judge")
    p.add_argument("--pipelines", default="single,single+verify,multi")
    p.add_argument("--runs", default="runs/notes")
    p.add_argument("--results", default="results/notes.json")
    p.add_argument("--fault-rate", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=60_000)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--score-only", action="store_true")
    p.set_defaults(func=cmd_notes)

    p = sub.add_parser("asr-eval", help="WER/CER of Whisper (optionally + LoRA adapter)")
    data_args(p)
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--adapter", default="", help="path to a PEFT LoRA adapter")
    p.add_argument("--max-utts", type=int, default=0, help="cap utterances (CPU subsets)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default="results/asr_eval.json")
    p.add_argument("--preds", default="", help="optional jsonl of per-utterance predictions")
    p.set_defaults(func=cmd_asr_eval)

    p = sub.add_parser("asr-compare", help="paired WER diff of two prediction files")
    p.add_argument("--a", required=True, help="preds jsonl (e.g. fine-tuned)")
    p.add_argument("--b", required=True, help="preds jsonl (e.g. baseline)")
    p.add_argument("--tokenizer", default="openai/whisper-small")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_asr_compare)

    p = sub.add_parser("asr-train", help="LoRA fine-tune Whisper on the train split")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--out", default="checkpoints/whisper-small-lora")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--max-dev-utts", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_asr_train)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
