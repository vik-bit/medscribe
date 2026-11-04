"""LLM-as-judge for claim support, independent of the citations a system emits.

For each claim we retrieve the two best-matching 4-utterance windows of the
transcript (lexical retrieval) and ask the judge whether the claim is supported
by them. The judge never sees which pipeline produced the claim. Claims the
judge skips are counted as unjudged and excluded from the rate.

Caveat (stated in the README): the judge is a small local model, and in our
runs it is the same model family as the generator.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from medscribe.agents.llm import LLM
from medscribe.agents.runtime import Budget, Context, Trace, call_typed
from medscribe.text import normalise_tokens

JUDGE_SYSTEM = (
    "You check whether statements in a medical note are supported by excerpts of the "
    "doctor-patient conversation. A statement is supported only if the excerpts say it "
    "(paraphrase is fine). If it adds details, numbers or conclusions that are not in the "
    "excerpts, it is not supported. Reply with JSON only."
)


class JudgeVerdict(BaseModel):
    id: int
    supported: bool


class JudgeOut(BaseModel):
    verdicts: list[JudgeVerdict]


def _top_windows(claim: str, turns: Sequence[str], k: int = 2, window: int = 4) -> list[int]:
    ct = set(normalise_tokens(claim))
    sets = [set(normalise_tokens(t)) for t in turns]
    scores = []
    for i in range(max(1, len(turns) - window + 1)):
        ev = set().union(*sets[i : i + window])
        scores.append((len(ct & ev), -i))
    scores.sort(reverse=True)
    starts: list[int] = []
    for _, neg_i in scores:
        i = -neg_i
        if all(abs(i - s) >= window for s in starts):
            starts.append(i)
        if len(starts) == k:
            break
    return starts


def judge_claims(
    llm: LLM, claims: Sequence[str], turns: Sequence[str], *, batch: int = 8
) -> tuple[list[bool | None], dict[str, int]]:
    """Return per-claim verdicts (None = unjudged) and token usage."""
    ctx = Context(llm=llm, budget=Budget(max_tokens=10**9, max_steps=10**6, max_seconds=10**6),
                  trace=Trace(), call_timeout_s=600)  # fmt: skip
    verdicts: list[bool | None] = [None] * len(claims)
    for b0 in range(0, len(claims), batch):
        blocks = []
        for j in range(b0, min(len(claims), b0 + batch)):
            ex = []
            for s in _top_windows(claims[j], turns):
                ex.append(" | ".join(turns[s : s + 4]))
            blocks.append(f"STATEMENT {j}: {claims[j]}\nEXCERPTS {j}: " + " || ".join(ex))
        user = "\n\n".join(blocks) + "\n\nFor every STATEMENT id above, say if it is supported."
        try:
            out = call_typed(ctx, "judge", JUDGE_SYSTEM, user, JudgeOut)
        except Exception:  # noqa: BLE001 - leave this batch unjudged
            continue
        for v in out.verdicts:
            if b0 <= v.id < min(len(claims), b0 + batch):
                verdicts[v.id] = v.supported
    usage = ctx.trace.per_agent().get("judge", {})
    return verdicts, {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }
