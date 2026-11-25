"""The multi-agent pipeline of ``orchestrator.run_multi``, re-implemented on LangGraph.

Same specialists, prompts and LLM-call layer (``specialists.py`` and
``runtime.call_typed``), so any difference from the hand-written version comes
from orchestration alone. LangGraph owns what a framework is for:

* **typed state**: a ``TypedDict`` of the pydantic messages passed between agents;
* **routing**: the planner's choices and the verifier-gated revise loop are
  conditional edges, not Python control flow;
* **checkpointing**: pass any LangGraph checkpointer (e.g. ``InMemorySaver``) and
  a ``thread_id``; every step's state is saved and can be inspected or resumed;
* **human-in-the-loop**: with ``human_review=True`` the graph pauses (via
  ``interrupt``) after verification and before the note is finalised. A reviewer
  resumes with ``"approve"`` or an edited note, and edits are re-verified;
* **tracing**: each node's wall time goes into the shared per-agent ``Trace``.
  LangSmith tracing is off unless you set ``LANGSMITH_TRACING=true`` and a key;
  nothing here needs an account.

The LLM client, budget and trace are run-scoped, non-serialisable objects, so
they travel in LangGraph's runtime ``context``, not in the checkpointed state.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from medscribe.agents import specialists as sp
from medscribe.agents.llm import LLM
from medscribe.agents.messages import (
    Entities,
    Plan,
    SoapNote,
    SpeakerMap,
    Transcript,
    VerifierReport,
)
from medscribe.agents.orchestrator import RunResult
from medscribe.agents.runtime import AgentFailed, Budget, BudgetExceeded, Context, Trace


class NoteState(TypedDict, total=False):
    transcript: Transcript
    plan: Plan
    speaker_map: SpeakerMap
    speaker_source: str
    work: Transcript  # transcript with roles applied
    entities: Entities
    note: SoapNote
    report: VerifierReport
    rounds: int
    rejected_per_round: list[int]
    revise_stopped: bool
    status: Literal["ok", "budget_exhausted", "failed"]
    error: str
    final_note: SoapNote
    dropped: int
    recited: int
    reviewed: str  # "", "approved" or "edited"


@dataclass
class GraphContext:
    ctx: Context  # LLM client + budget + trace, shared with the specialists
    max_rounds: int = 2


def _timed(name: str, fn):
    """Wrap a node so its wall time is recorded in the shared trace."""

    def node(state: NoteState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
        t0 = time.perf_counter()
        out = fn(state, runtime.context)
        runtime.context.ctx.trace.add(
            "graph", "info", detail=f"node {name} {time.perf_counter() - t0:.3f}s"
        )
        return out

    return node


# --- nodes ------------------------------------------------------------------
def _planner(state: NoteState, g: GraphContext) -> dict[str, Any]:
    p = sp.plan(g.ctx, state["transcript"])  # falls back to the default plan itself
    g.ctx.trace.add("planner", "info", detail=f"steps={p.steps}")
    return {"plan": p, "rounds": 0, "rejected_per_round": [], "status": "ok", "error": ""}


def _speaker(state: NoteState, g: GraphContext) -> dict[str, Any]:
    t = state["transcript"]
    if "speaker" in state["plan"].steps:
        m, source = sp.label_speakers(g.ctx, t)
    else:
        m, source = sp.heuristic_speaker_map(t), "heuristic"
    return {"speaker_map": m, "speaker_source": source, "work": sp.relabel(t, m)}


def _extractor(state: NoteState, g: GraphContext) -> dict[str, Any]:
    if "extractor" in state["plan"].steps:
        return {"entities": sp.extract(g.ctx, state["work"])}
    return {"entities": Entities(entities=[])}


def _writer(state: NoteState, g: GraphContext) -> dict[str, Any]:
    try:
        return {"note": sp.write_note(g.ctx, state["work"], state["entities"], state["plan"])}
    except BudgetExceeded as e:
        return {"status": "budget_exhausted", "error": str(e)}
    except AgentFailed as e:
        return {"status": "failed", "error": str(e)}


def _verify(state: NoteState, g: GraphContext) -> dict[str, Any]:
    report = sp.verify(state["work"], state["note"])
    return {
        "report": report,
        "rejected_per_round": [*state["rejected_per_round"], len(report.rejected)],
    }


def _revise(state: NoteState, g: GraphContext) -> dict[str, Any]:
    try:
        note = sp.revise(g.ctx, state["work"], state["note"], state["report"])
    except BudgetExceeded as e:
        return {"status": "budget_exhausted", "error": str(e), "revise_stopped": True}
    except AgentFailed as e:
        g.ctx.trace.add("writer", "info", detail=f"revision failed: {e}")
        return {"revise_stopped": True}
    return {"note": note, "rounds": state["rounds"] + 1}


def _human_review(state: NoteState, g: GraphContext) -> dict[str, Any]:
    """Pause for a reviewer. Resume with "approve" or a SoapNote (dict) to replace it."""
    decision = interrupt(
        {
            "note": state["note"].model_dump(),
            "rejected": [v.model_dump() for v in state["report"].rejected],
            "hint": 'resume with Command(resume="approve") or Command(resume=<edited note dict>)',
        }
    )
    if decision == "approve":
        return {"reviewed": "approved"}
    edited = SoapNote.model_validate(decision)
    # Human edits go through the same grounding check as model output.
    return {"note": edited, "report": sp.verify(state["work"], edited), "reviewed": "edited"}


def _finalize(state: NoteState, g: GraphContext) -> dict[str, Any]:
    report = state["report"]
    dropped = len(report.rejected)
    recited = sum(v.recited for v in report.verdicts)
    g.ctx.trace.add(
        "verifier",
        "info",
        detail=f"dropped {dropped}, re-cited {recited}, after {state['rounds']} rounds",
    )
    return {"final_note": sp.drop_rejected(state["note"], report), "dropped": dropped,
            "recited": recited}  # fmt: skip


# --- routing ----------------------------------------------------------------
def _after_writer(state: NoteState) -> str:
    return "verify" if state["status"] == "ok" else END


def _make_after_verify(human_review: bool):
    def route(state: NoteState, runtime: Runtime[GraphContext]) -> str:
        more = (
            state["report"].rejected
            and state["rounds"] < runtime.context.max_rounds
            and not state.get("revise_stopped")
        )
        if more:
            return "revise"
        return "human_review" if human_review else "finalize"

    return route


def _after_revise(state: NoteState) -> str:
    # A failed revision keeps the previous note and report, like the hand-written loop.
    return "finalize" if state.get("revise_stopped") else "verify"


def build_graph(*, human_review: bool = False, checkpointer: Any = None):
    """Compile the note-writing graph. ``human_review`` needs a checkpointer."""
    if human_review and checkpointer is None:
        raise ValueError("human_review=True needs a checkpointer to pause and resume")
    g = StateGraph(NoteState, context_schema=GraphContext)
    for name, fn in [("planner", _planner), ("speaker", _speaker), ("extractor", _extractor),
                     ("writer", _writer), ("verify", _verify), ("revise", _revise),
                     ("finalize", _finalize)]:  # fmt: skip
        g.add_node(name, _timed(name, fn))
    g.add_edge(START, "planner")
    g.add_edge("planner", "speaker")
    g.add_edge("speaker", "extractor")
    g.add_edge("extractor", "writer")
    g.add_conditional_edges("writer", _after_writer, ["verify", END])
    targets = ["revise", "human_review" if human_review else "finalize"]
    g.add_conditional_edges("verify", _make_after_verify(human_review), targets)
    g.add_conditional_edges("revise", _after_revise, ["verify", "finalize"])
    if human_review:
        g.add_node("human_review", _timed("human_review", _human_review))
        g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def _result(state: dict[str, Any], ctx: Context, budget: Budget, name: str) -> RunResult:
    status = state.get("status", "ok")
    note = state.get("final_note") or SoapNote()
    return RunResult(
        name,
        note,
        status,
        ctx.trace,
        budget,
        speaker_map=state.get("speaker_map"),
        speaker_source=state.get("speaker_source", ""),
        rounds=state.get("rounds", 0),
        rejected_per_round=list(state.get("rejected_per_round", [])),
        dropped=state.get("dropped", 0),
        recited=state.get("recited", 0),
        error=state.get("error", ""),
    )


def run_multi_langgraph(llm: LLM, t: Transcript, *, budget: Budget | None = None,
                        max_rounds: int = 2, call_timeout_s: float = 600.0,
                        checkpointer: Any = None, thread_id: str | None = None,
                        graph: Any = None) -> RunResult:  # fmt: skip
    """Drop-in replacement for ``orchestrator.run_multi`` (same arguments and result)."""
    budget = budget or Budget()
    ctx = Context(llm=llm, budget=budget, trace=Trace(), call_timeout_s=call_timeout_s)
    graph = graph or build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    state = graph.invoke({"transcript": t}, config, context=GraphContext(ctx, max_rounds))
    name = "multi-langgraph" if checkpointer is None else "multi-langgraph+ckpt"
    return _result(state, ctx, budget, name)


@dataclass
class ReviewSession:
    """A paused run waiting for a human decision (``human_review=True``)."""

    graph: Any
    config: dict[str, Any]
    context: GraphContext
    budget: Budget
    pending: dict[str, Any]  # interrupt payload: note, rejected claims, hint

    def resume(self, decision: str | dict[str, Any]) -> RunResult:
        state = self.graph.invoke(Command(resume=decision), self.config, context=self.context)
        return _result(state, self.context.ctx, self.budget, "multi-langgraph+review")


def start_review(llm: LLM, t: Transcript, checkpointer: Any, *, budget: Budget | None = None,
                 max_rounds: int = 2, call_timeout_s: float = 600.0,
                 thread_id: str | None = None) -> ReviewSession | RunResult:  # fmt: skip
    """Run until the human-review interrupt. Returns a ``RunResult`` if the run
    ended before review (e.g. the writer failed), else a ``ReviewSession``."""
    budget = budget or Budget()
    ctx = Context(llm=llm, budget=budget, trace=Trace(), call_timeout_s=call_timeout_s)
    graph = build_graph(human_review=True, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    gctx = GraphContext(ctx, max_rounds)
    state = graph.invoke({"transcript": t}, config, context=gctx)
    pending = state.get("__interrupt__")
    if not pending:
        return _result(state, ctx, budget, "multi-langgraph+review")
    return ReviewSession(graph, config, gctx, budget, pending[0].value)
