"""Note-quality metrics against PriMock57's clinician-written notes.

* ``note_f1``          token-level F1 (multiset of normalised content tokens)
                       between the generated note and the clinician note.
* ``highlight_recall`` fraction of the clinician's highlighted key facts whose
                       content tokens are at least half present in the note.
* ``ungrounded_rate``  fraction of generated claims whose best 4-utterance
                       window in the transcript covers < 50% of the claim's
                       content tokens, or that state a number never said
                       nearby. Lexical and citation-free; see README caveats -
                       it is related to the verifier's own check, so the LLM
                       judge (``judge.py``) is reported alongside it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from medscribe.grounding import best_window_support, unsupported_numbers
from medscribe.text import normalise_tokens


def token_f1(pred: str, ref: str) -> tuple[float, float, float]:
    p, r = Counter(normalise_tokens(pred)), Counter(normalise_tokens(ref))
    overlap = sum((p & r).values())
    if not p or not r or not overlap:
        return 0.0, 0.0, 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(r.values())
    return prec, rec, 2 * prec * rec / (prec + rec)


def highlight_recall(note: str, highlights: Sequence[str], min_cover: float = 0.5) -> float:
    if not highlights:
        return float("nan")
    note_toks = set(normalise_tokens(note))
    hit = 0
    for h in highlights:
        ht = set(normalise_tokens(h))
        if ht and len(ht & note_toks) / len(ht) >= min_cover:
            hit += 1
    return hit / len(highlights)


def claim_grounded(
    claim: str, turns: Sequence[str], window: int = 4, threshold: float = 0.5
) -> bool:
    score, start = best_window_support(claim, turns, window)
    if score < threshold:
        return False
    lo, hi = max(0, start - window), min(len(turns), start + 2 * window)
    return not unsupported_numbers(claim, turns[lo:hi])


def ungrounded_rate(claims: Sequence[str], turns: Sequence[str]) -> float:
    if not claims:
        return float("nan")
    return sum(not claim_grounded(c, turns) for c in claims) / len(claims)


def split_reference_claims(note: str) -> list[str]:
    """Sentence/line split of a free-text clinician note (to score it as a 'system')."""
    import re

    parts = re.split(r"(?<=[.!?])\s+|\n+", note)
    return [p.strip(" -•\t") for p in parts if len(normalise_tokens(p)) >= 2]
