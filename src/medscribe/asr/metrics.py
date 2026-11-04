"""WER / CER with consultation-level (cluster) bootstrap confidence intervals.

Utterances from the same consultation share a speaker pair, microphone and
topic, so they are not independent. Resampling whole consultations gives
wider - more honest - intervals than resampling utterances.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import jiwer
import numpy as np


@dataclass(frozen=True)
class UttScore:
    cid: str
    word_errors: int
    ref_words: int
    char_errors: int
    ref_chars: int


def score_utterance(cid: str, ref: str, hyp: str) -> UttScore | None:
    """Edit counts for one (already normalised) pair; None if the reference is empty."""
    if not ref.strip():
        return None
    w = jiwer.process_words(ref, hyp if hyp.strip() else "<empty>")
    c = jiwer.process_characters(ref, hyp if hyp.strip() else " ")
    we = w.substitutions + w.deletions + w.insertions
    ce = c.substitutions + c.deletions + c.insertions
    if not hyp.strip():  # an empty hypothesis deletes every reference word
        we, ce = len(ref.split()), len(ref)
    return UttScore(cid, we, len(ref.split()), ce, len(ref))


def corpus_rates(scores: Sequence[UttScore]) -> tuple[float, float]:
    w = sum(s.word_errors for s in scores) / max(1, sum(s.ref_words for s in scores))
    c = sum(s.char_errors for s in scores) / max(1, sum(s.ref_chars for s in scores))
    return w, c


def _by_cid(scores: Sequence[UttScore]) -> dict[str, np.ndarray]:
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for s in scores:
        a = agg[s.cid]
        a[0] += s.word_errors
        a[1] += s.ref_words
        a[2] += s.char_errors
        a[3] += s.ref_chars
    return {k: np.array(v, dtype=float) for k, v in agg.items()}


def cluster_bootstrap(
    scores: Sequence[UttScore], *, n: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Corpus WER/CER plus 95% CIs, resampling consultations with replacement."""
    per = _by_cid(scores)
    mat = np.stack(list(per.values()))  # [n_cid, 4]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(mat), size=(n, len(mat)))
    tot = mat[idx].sum(axis=1)  # [n, 4]
    wer_s, cer_s = tot[:, 0] / tot[:, 1], tot[:, 2] / tot[:, 3]
    wer, cer = corpus_rates(scores)
    return {
        "wer": wer,
        "wer_lo": float(np.quantile(wer_s, 0.025)),
        "wer_hi": float(np.quantile(wer_s, 0.975)),
        "cer": cer,
        "cer_lo": float(np.quantile(cer_s, 0.025)),
        "cer_hi": float(np.quantile(cer_s, 0.975)),
        "n_utts": len(scores),
        "n_consultations": len(mat),
        "ref_words": int(mat[:, 1].sum()),
    }


def paired_cluster_bootstrap(
    a: Sequence[UttScore], b: Sequence[UttScore], *, n: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """WER(a) - WER(b) on the same utterances, with a consultation-level CI."""
    pa, pb = _by_cid(a), _by_cid(b)
    keys = sorted(set(pa) & set(pb))
    ma, mb = np.stack([pa[k] for k in keys]), np.stack([pb[k] for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    ta, tb = ma[idx].sum(axis=1), mb[idx].sum(axis=1)
    d = ta[:, 0] / ta[:, 1] - tb[:, 0] / tb[:, 1]
    point = ma[:, 0].sum() / ma[:, 1].sum() - mb[:, 0].sum() / mb[:, 1].sum()
    return {
        "wer_diff": float(point),
        "lo": float(np.quantile(d, 0.025)),
        "hi": float(np.quantile(d, 0.975)),
        "p_ge_0": float((d >= 0).mean()),
        "n_consultations": len(keys),
    }
