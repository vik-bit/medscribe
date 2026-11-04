"""Evaluate Whisper (base or + LoRA adapter) on a PriMock57 split.

CPU-friendly: ``--max-utts`` evaluates a fixed, seeded subset so a laptop
without a GPU can reproduce the numbers in ``results/``.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from medscribe.asr.metrics import UttScore, cluster_bootstrap, score_utterance
from medscribe.asr.segments import Segment, load_audio, split_segments

SUBSET_SEED = 0


def subset(segs: list[Segment], max_utts: int) -> list[Segment]:
    if not max_utts or max_utts >= len(segs):
        return segs
    picked = random.Random(SUBSET_SEED).sample(range(len(segs)), max_utts)
    return [segs[i] for i in sorted(picked)]


def load_model(model_name: str, adapter: str = ""):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.to(device).eval()
    return processor, model, device


def normaliser(processor):
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)


def transcribe(processor, model, device, segs: list[Segment], batch_size: int = 8, log=print):
    import torch

    hyps: list[str] = []
    t0 = time.perf_counter()
    for i in range(0, len(segs), batch_size):
        batch = segs[i : i + batch_size]
        feats = processor.feature_extractor(
            [load_audio(s) for s in batch], sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(device, dtype=model.dtype)
        with torch.inference_mode():
            ids = model.generate(feats, language="en", task="transcribe", max_new_tokens=225)
        hyps.extend(processor.batch_decode(ids, skip_special_tokens=True))
        if (i // batch_size) % 10 == 0:
            log(f"  {i + len(batch)}/{len(segs)} utts, {time.perf_counter() - t0:.0f}s")
    return hyps


def score_all(norm, segs: list[Segment], hyps: list[str]) -> list[UttScore]:
    out = []
    for s, h in zip(segs, hyps, strict=True):
        sc = score_utterance(s.cid, norm(s.text), norm(h))
        if sc is not None:
            out.append(sc)
    return out


def compare(args) -> None:
    """Paired WER difference between two prediction files on the same utterances."""
    from transformers import WhisperTokenizer
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    from medscribe.asr.metrics import paired_cluster_bootstrap

    norm = EnglishTextNormalizer(
        WhisperTokenizer.from_pretrained(args.tokenizer).english_spelling_normalizer
    )

    def load(p):
        return {
            r["key"]: r for r in map(json.loads, Path(p).read_text(encoding="utf-8").splitlines())
        }

    a, b = load(args.a), load(args.b)
    keys = sorted(set(a) & set(b))
    sa = [score_utterance(a[k]["cid"], norm(a[k]["ref"]), norm(a[k]["hyp"])) for k in keys]
    sb = [score_utterance(b[k]["cid"], norm(b[k]["ref"]), norm(b[k]["hyp"])) for k in keys]
    pairs = [(x, y) for x, y in zip(sa, sb, strict=True) if x is not None and y is not None]
    res = paired_cluster_bootstrap([x for x, _ in pairs], [y for _, y in pairs])
    res.update(a=args.a, b=args.b, n_utts=len(pairs),
               wer_a=cluster_bootstrap([x for x, _ in pairs]), wer_b=cluster_bootstrap([y for _, y in pairs]))  # fmt: skip
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def main(args) -> None:
    segs = subset(split_segments(Path(args.data), args.split), args.max_utts)
    print(
        f"{len(segs)} utterances from split={args.split}, {sum(s.duration for s in segs) / 60:.1f} min"
    )
    processor, model, device = load_model(args.model, args.adapter)
    t0 = time.perf_counter()
    hyps = transcribe(processor, model, device, segs, args.batch_size)
    elapsed = time.perf_counter() - t0
    norm = normaliser(processor)
    res = cluster_bootstrap(score_all(norm, segs, hyps))
    res.update(model=args.model, adapter=args.adapter or None, split=args.split, device=device,
               max_utts=args.max_utts, subset_seed=SUBSET_SEED, audio_min=sum(s.duration for s in segs) / 60,
               decode_s=elapsed)  # fmt: skip
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    if args.preds:
        with open(args.preds, "w", encoding="utf-8") as f:
            for s, h in zip(segs, hyps, strict=True):
                f.write(json.dumps({"key": s.key, "cid": s.cid, "ref": s.text, "hyp": h}) + "\n")
    print(json.dumps(res, indent=1))
