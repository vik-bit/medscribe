"""Error breakdown of an ``asr-eval --preds`` file.

Reports WER per channel (doctor / patient), the most frequent substitutions,
deletions and insertions, and WER when ``ok``/``okay`` are treated as the same
word (PriMock57 transcribers write "OK", Whisper writes "Okay", and the
standard Whisper normaliser does not merge them).

    python scripts/asr_error_analysis.py results/asr_test300_base_cpu.jsonl --out results/x.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import jiwer
from transformers import WhisperTokenizer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from medscribe.asr.metrics import cluster_bootstrap, score_utterance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--tokenizer", default="openai/whisper-small")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    tok = WhisperTokenizer.from_pretrained(args.tokenizer)
    norm = EnglishTextNormalizer(tok.english_spelling_normalizer)
    rows = [json.loads(x) for x in Path(args.preds).read_text(encoding="utf-8").splitlines()]

    sub, dele, ins = collections.Counter(), collections.Counter(), collections.Counter()
    per_channel: dict[str, list] = collections.defaultdict(list)
    merged = []
    for r in rows:
        ref, hyp = norm(r["ref"]), norm(r["hyp"])
        s = score_utterance(r["cid"], ref, hyp)
        if s is None:
            continue
        per_channel[r["key"].split(":")[1]].append(s)
        m = score_utterance(r["cid"], ref.replace("okay", "ok"), hyp.replace("okay", "ok"))
        merged.append(m)
        if not hyp.strip():
            continue
        out = jiwer.process_words(ref, hyp)
        rw, hw = ref.split(), hyp.split()
        for a in out.alignments[0]:
            if a.type == "substitute":
                for i, j in zip(
                    range(a.ref_start_idx, a.ref_end_idx),
                    range(a.hyp_start_idx, a.hyp_end_idx),
                    strict=True,
                ):
                    sub[f"{rw[i]} -> {hw[j]}"] += 1
            elif a.type == "delete":
                dele.update(rw[a.ref_start_idx : a.ref_end_idx])
            elif a.type == "insert":
                ins.update(hw[a.hyp_start_idx : a.hyp_end_idx])

    res = {
        "preds": args.preds,
        "per_channel": {k: cluster_bootstrap(v) for k, v in per_channel.items()},
        "ok_okay_merged": cluster_bootstrap(merged),
        "errors": {
            "substitutions": sum(sub.values()),
            "deletions": sum(dele.values()),
            "insertions": sum(ins.values()),
        },  # fmt: skip
        "top_substitutions": sub.most_common(15),
        "top_deletions": dele.most_common(15),
        "top_insertions": ins.most_common(10),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
