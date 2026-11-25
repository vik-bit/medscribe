"""LangGraph orchestrator: behavioural equivalence with run_multi, checkpoints, review."""

import copy

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from medscribe.agents.langgraph_orchestrator import (  # noqa: E402
    ReviewSession,
    build_graph,
    run_multi_langgraph,
    start_review,
)
from medscribe.agents.llm import FaultInjectingLLM, ScriptedLLM  # noqa: E402
from medscribe.agents.orchestrator import run_multi  # noqa: E402
from medscribe.agents.runtime import Budget  # noqa: E402
from tests.conftest import GOOD_NOTE, scripted_responder  # noqa: E402

BAD = copy.deepcopy(GOOD_NOTE)
BAD["subjective"].append({"text": "Cough for nine days", "evidence": [3]})
BAD["plan"].append({"text": "Start amoxicillin", "evidence": [8]})
FIX = {"revised": [{"section": "subjective", "index": 3, "action": "rewrite",
                    "text": "Cough for about five days", "evidence": [3]}]}  # fmt: skip


def _writer_down(system, user):
    if "SOAP" in user and "Plan the documentation" not in user:
        raise TimeoutError("writer down")
    return scripted_responder()(system, user)


def _no_speaker_plan(system, user):
    if "Plan the documentation work" in user:
        return '{"steps": ["writer"], "presenting_complaint": "cough"}'
    return scripted_responder()(system, user)


SCENARIOS = {
    "happy": (lambda gold: scripted_responder(speaker=gold), {}, 0.0),
    "revise_then_drop": (
        lambda gold: scripted_responder(note=BAD, speaker=gold, revision=FIX),
        {},
        0.0,
    ),
    "budget_exhausted": (
        lambda gold: scripted_responder(note=BAD, speaker=gold),
        {"max_steps": 4},
        0.0,
    ),
    "writer_failure": (lambda gold: _writer_down, {}, 0.0),
    "planner_skips_specialists": (lambda gold: _no_speaker_plan, {}, 0.0),
    "fault_injection": (lambda gold: scripted_responder(speaker=gold), {"max_steps": 20}, 0.5),
}


def _run(fn, scenario, transcript):
    t, gold = transcript
    make, budget_kw, fault = SCENARIOS[scenario]
    inner = ScriptedLLM(make(gold))
    llm = FaultInjectingLLM(inner, rate=fault, seed=3) if fault else inner
    return fn(llm, t, budget=Budget(**budget_kw)), inner


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_langgraph_matches_hand_written_orchestrator(transcript, scenario):
    a, llm_a = _run(run_multi, scenario, transcript)
    b, llm_b = _run(run_multi_langgraph, scenario, transcript)
    assert llm_a.calls == llm_b.calls  # same prompts, same order, same count
    assert b.note == a.note
    for field in ("status", "rounds", "rejected_per_round", "dropped", "recited",
                  "speaker_map", "speaker_source", "error"):  # fmt: skip
        assert getattr(b, field) == getattr(a, field), field
    sa, sb = a.summary(), b.summary()
    for k in ("llm_calls", "llm_failures", "prompt_tokens", "completion_tokens"):
        assert sa[k] == sb[k], k


def test_checkpointer_records_every_step(transcript):
    t, gold = transcript
    saver = InMemorySaver()
    r = run_multi_langgraph(ScriptedLLM(scripted_responder(note=BAD, speaker=gold, revision=FIX)),
                            t, checkpointer=saver, thread_id="t1")  # fmt: skip
    assert r.status == "ok"
    graph = build_graph(checkpointer=saver)
    history = list(graph.get_state_history({"configurable": {"thread_id": "t1"}}))
    # newest first; each checkpoint knows which node(s) run next
    order = [h.next for h in reversed(history) if h.next]
    flat = [n for step in order for n in step]
    assert flat[:5] == ["__start__", "planner", "speaker", "extractor", "writer"]
    assert flat.count("revise") == 2 and flat[-1] == "finalize"
    final = history[0].values
    assert final["rounds"] == 2 and final["final_note"] == r.note


def test_node_timings_are_traced(transcript):
    t, gold = transcript
    r = run_multi_langgraph(ScriptedLLM(scripted_responder(speaker=gold)), t)
    nodes = [e.detail.split()[1] for e in r.trace.events if e.agent == "graph"]
    assert nodes == ["planner", "speaker", "extractor", "writer", "verify", "finalize"]


def test_human_review_requires_checkpointer():
    with pytest.raises(ValueError):
        build_graph(human_review=True)


def test_human_review_approve(transcript):
    t, gold = transcript
    s = start_review(ScriptedLLM(scripted_responder(speaker=gold)), t, InMemorySaver())
    assert isinstance(s, ReviewSession)
    assert s.pending["note"] == GOOD_NOTE and s.pending["rejected"] == []
    r = s.resume("approve")
    assert r.status == "ok" and len(r.note.claims()) == 5


def test_human_review_edits_are_reverified(transcript):
    t, gold = transcript
    s = start_review(ScriptedLLM(scripted_responder(speaker=gold)), t, InMemorySaver())
    edited = copy.deepcopy(GOOD_NOTE)
    edited["assessment"] = [{"text": "Bacterial pneumonia needing antibiotics", "evidence": [8]}]
    r = s.resume(edited)
    texts = [c.text for _, c in r.note.claims()]
    assert "Bacterial pneumonia needing antibiotics" not in texts  # ungrounded edit dropped
    assert r.dropped == 1


def test_review_not_reached_when_writer_fails(transcript):
    t, _ = transcript
    out = start_review(ScriptedLLM(_writer_down), t, InMemorySaver())
    assert not isinstance(out, ReviewSession) and out.status == "failed"
