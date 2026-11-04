"""Lexical grounding: how much of a claim is literally supported by transcript text."""

from __future__ import annotations

from collections.abc import Sequence

from medscribe.text import normalise_tokens, numbers_in


def support(claim: str, evidence_texts: Sequence[str]) -> float:
    """Fraction of the claim's content tokens that occur in the evidence text."""
    claim_toks = set(normalise_tokens(claim))
    if not claim_toks:
        return 0.0
    ev = set()
    for t in evidence_texts:
        ev.update(normalise_tokens(t))
    return len(claim_toks & ev) / len(claim_toks)


def unsupported_numbers(claim: str, evidence_texts: Sequence[str]) -> set[str]:
    """Numbers in the claim that never appear in the evidence (e.g. a made-up dose)."""
    ev: set[str] = set()
    for t in evidence_texts:
        ev |= numbers_in(t)
    return numbers_in(claim) - ev


def best_window_support(claim: str, turns: Sequence[str], window: int = 4) -> tuple[float, int]:
    """Max support over all windows of ``window`` consecutive turns; returns (score, start)."""
    if not turns:
        return 0.0, 0
    claim_toks = set(normalise_tokens(claim))
    if not claim_toks:
        return 0.0, 0
    tok_sets = [set(normalise_tokens(t)) for t in turns]
    best, best_i = 0.0, 0
    for i in range(max(1, len(turns) - window + 1)):
        ev = set().union(*tok_sets[i : i + window])
        s = len(claim_toks & ev) / len(claim_toks)
        if s > best:
            best, best_i = s, i
    return best, best_i
