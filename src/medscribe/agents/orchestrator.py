"""Pipelines compared in the evaluation.

* ``single``         - one LLM call: transcript -> SOAP note.
* ``single+verify``  - same call, then the deterministic verifier drops
                       ungrounded claims (ablation: is any gain just filtering?).
* ``multi``          - planner -> speaker labelling -> extractor -> writer ->
                       verifier -> bounded revise loop -> drop what still fails.

All three see the same input: a diarized transcript whose speakers are the
anonymous clusters SPEAKER_A / SPEAKER_B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medscribe.agents import specialists as sp
from medscribe.agents.llm import LLM
from medscribe.agents.messages import SoapNote, SpeakerMap, Transcript
from medscribe.agents.runtime import AgentFailed, Budget, BudgetExceeded, Context, Trace, call_typed


@dataclass
class RunResult:
    pipeline: str
    note: SoapNote
    status: str  # ok | budget_exhausted | failed
    trace: Trace
    budget: Budget
    speaker_map: SpeakerMap | None = None
    speaker_source: str = ""
    rounds: int = 0
    rejected_per_round: list[int] = field(default_factory=list)
    dropped: int = 0
    recited: int = 0
    error: str = ""

    def summary(self) -> dict[str, Any]:
        agents = self.trace.per_agent()
        return {
            "pipeline": self.pipeline,
            "status": self.status,
            "claims": len(self.note.claims()),
            "rounds": self.rounds,
            "rejected_per_round": self.rejected_per_round,
            "dropped": self.dropped,
            "recited": self.recited,
            "llm_calls": sum(a["calls"] for a in agents.values()),
            "llm_failures": sum(a["failures"] for a in agents.values()),
            "prompt_tokens": sum(a["prompt_tokens"] for a in agents.values()),
            "completion_tokens": sum(a["completion_tokens"] for a in agents.values()),
            "latency_s": round(self.budget.elapsed(), 2),
            "speaker_map": self.speaker_map.model_dump() if self.speaker_map else None,
            "speaker_source": self.speaker_source,
            "per_agent": agents,
            "error": self.error,
        }


SINGLE_TASK = (
    "The speakers were separated by a diarization system into SPEAKER_A and SPEAKER_B; "
    "work out which is the doctor. " + sp.NOTE_TASK
)


def run_single(llm: LLM, t: Transcript, *, verify: bool = False, budget: Budget | None = None,
               call_timeout_s: float = 600.0) -> RunResult:  # fmt: skip
    budget = budget or Budget()
    ctx = Context(llm=llm, budget=budget, trace=Trace(), call_timeout_s=call_timeout_s)
    name = "single+verify" if verify else "single"
    try:
        note = call_typed(ctx, "single", sp.SYSTEM, sp.transcript_block(t) + SINGLE_TASK, SoapNote)
    except (AgentFailed, BudgetExceeded) as e:
        return RunResult(name, SoapNote(), "failed", ctx.trace, budget, error=str(e))
    dropped = recited = 0
    rejected: list[int] = []
    if verify:
        report = sp.verify(t, note)
        ctx.trace.add("verifier", "info", detail=f"{len(report.rejected)} rejected")
        rejected = [len(report.rejected)]
        dropped = len(report.rejected)
        recited = sum(v.recited for v in report.verdicts)
        note = sp.drop_rejected(note, report)
    return RunResult(name, note, "ok", ctx.trace, budget, rejected_per_round=rejected,
                     dropped=dropped, recited=recited)  # fmt: skip


def run_multi(llm: LLM, t: Transcript, *, budget: Budget | None = None, max_rounds: int = 2,
              call_timeout_s: float = 600.0) -> RunResult:  # fmt: skip
    """Planner-driven multi-agent pipeline with a verifier-gated revise loop.

    Degradation policy (each step is recorded in the trace):
      planner fails     -> default plan
      speaker fails     -> question-count heuristic
      extractor fails   -> writer runs without entities
      writer fails      -> status=failed, empty note
      budget exhausted  -> stop revising, keep only verified claims
    """
    budget = budget or Budget()
    ctx = Context(llm=llm, budget=budget, trace=Trace(), call_timeout_s=call_timeout_s)
    res = RunResult("multi", SoapNote(), "ok", ctx.trace, budget)
    try:
        p = sp.plan(ctx, t)
        ctx.trace.add("planner", "info", detail=f"steps={p.steps}")
        work = t
        if "speaker" in p.steps:
            res.speaker_map, res.speaker_source = sp.label_speakers(ctx, t)
        else:
            res.speaker_map, res.speaker_source = sp.heuristic_speaker_map(t), "heuristic"
        work = sp.relabel(t, res.speaker_map)
        entities = sp.extract(ctx, work) if "extractor" in p.steps else sp.Entities(entities=[])
        note = sp.write_note(ctx, work, entities, p)
    except BudgetExceeded as e:
        res.status, res.error = "budget_exhausted", str(e)
        return res
    except AgentFailed as e:
        res.status, res.error = "failed", str(e)
        return res

    report = sp.verify(work, note)
    res.rejected_per_round.append(len(report.rejected))
    while report.rejected and res.rounds < max_rounds:
        try:
            note = sp.revise(ctx, work, note, report)
        except BudgetExceeded as e:
            res.status, res.error = "budget_exhausted", str(e)
            break
        except AgentFailed as e:
            ctx.trace.add("writer", "info", detail=f"revision failed: {e}")
            break
        res.rounds += 1
        report = sp.verify(work, note)
        res.rejected_per_round.append(len(report.rejected))
    res.dropped = len(report.rejected)
    res.recited = sum(v.recited for v in report.verdicts)
    res.note = sp.drop_rejected(note, report)
    ctx.trace.add(
        "verifier",
        "info",
        detail=f"dropped {res.dropped}, re-cited {res.recited}, after {res.rounds} rounds",
    )
    return res
