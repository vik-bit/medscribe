import copy
import json

import pytest

from medscribe.agents.llm import FaultInjectingLLM, ScriptedLLM
from medscribe.agents.messages import SoapNote, Transcript
from medscribe.agents.orchestrator import run_multi, run_single
from medscribe.agents.runtime import (
    AgentFailed,
    Budget,
    BudgetExceeded,
    Context,
    Trace,
    call_typed,
    parse_model,
)
from medscribe.agents.specialists import apply_revision, heuristic_speaker_map, verify
from tests.conftest import GOOD_NOTE, scripted_responder


def _t(transcript) -> Transcript:
    return transcript[0]


# --- runtime ----------------------------------------------------------------
def test_parse_model_accepts_fenced_json():
    s = parse_model('```json\n{"subjective": [{"text": "x", "evidence": [1]}]}\n```', SoapNote)
    assert s.subjective[0].evidence == [1]


def test_parse_retry_is_charged_to_budget():
    replies = iter(["not json", json.dumps({"SPEAKER_A": "DOCTOR", "SPEAKER_B": "PATIENT"})])
    llm = ScriptedLLM(lambda s, u: next(replies))
    ctx = Context(llm=llm, budget=Budget(), trace=Trace(), parse_retries=1)
    from medscribe.agents.messages import SpeakerMap

    out = call_typed(ctx, "speaker", "sys", "user", SpeakerMap)
    assert out.SPEAKER_A == "DOCTOR"
    assert ctx.budget.steps == 2
    assert [e.event for e in ctx.trace.events if e.event != "call"] == ["parse_error", "ok"]
    assert "not valid JSON" in llm.calls[1][1]


def test_budget_blocks_calls():
    b = Budget(max_steps=0)
    with pytest.raises(BudgetExceeded):
        b.check()


# --- verifier -------------------------------------------------------------
def test_verifier_accepts_grounded_note(transcript):
    report = verify(_t(transcript), SoapNote.model_validate(GOOD_NOTE))
    assert report.rejected == [], [v.reason for v in report.rejected]


@pytest.mark.parametrize(
    "claim,reason",
    [
        ({"text": "Cough for five days", "evidence": [99]}, "no valid"),
        ({"text": "Cough for nine days", "evidence": [1, 2, 3]}, "numbers"),
        ({"text": "History of type 2 diabetes", "evidence": [1]}, "numbers"),
        ({"text": "Penicillin allergy with rash", "evidence": [7]}, "only"),
    ],
)
def test_verifier_rejects_ungrounded(transcript, claim, reason):
    report = verify(_t(transcript), SoapNote(subjective=[claim]), recite=False)
    (v,) = report.rejected
    assert reason in v.reason


def test_verifier_recites_supported_claim_with_wrong_ids(transcript):
    t = _t(transcript)
    note = SoapNote(subjective=[{"text": "Cough for five days", "evidence": [99]}])
    (v,) = verify(t, note).verdicts
    assert v.grounded and v.recited and set(v.evidence) >= {1, 3}
    from medscribe.agents.specialists import apply_verdicts

    assert apply_verdicts(note, verify(t, note)).subjective[0].evidence == v.evidence


@pytest.mark.parametrize("text", ["Cough for nine days", "Penicillin allergy with rash"])
def test_recitation_never_rescues_unsupported_claims(transcript, text):
    note = SoapNote(subjective=[{"text": text, "evidence": [1]}])
    assert len(verify(_t(transcript), note).rejected) == 1


def test_apply_revision_rewrites_and_drops():
    note = SoapNote.model_validate(GOOD_NOTE)
    rev = {
        "revised": [
            {
                "section": "subjective",
                "index": 0,
                "action": "rewrite",
                "text": "Cough",
                "evidence": [1],
            },
            {"section": "plan", "index": 0, "action": "drop"},
            {"section": "plan", "index": 7, "action": "drop"},
        ]
    }  # out of range: ignored
    from medscribe.agents.messages import Revision

    out = apply_revision(note, Revision.model_validate(rev))
    assert out.subjective[0].text == "Cough" and out.plan == []
    assert len(out.subjective) == 3


def test_heuristic_speaker_map_picks_question_asker(transcript):
    t, gold = transcript
    m = heuristic_speaker_map(t)
    assert {"SPEAKER_A": m.SPEAKER_A, "SPEAKER_B": m.SPEAKER_B} == gold


# --- pipelines ------------------------------------------------------------
def test_multi_pipeline_happy_path(transcript):
    t, gold = transcript
    llm = ScriptedLLM(scripted_responder(speaker=gold))
    r = run_multi(llm, t)
    assert r.status == "ok" and r.dropped == 0 and r.rounds == 0
    assert len(r.note.claims()) == 5
    assert r.speaker_source == "llm"
    agents = r.trace.per_agent()
    assert set(agents) >= {"planner", "speaker", "extractor", "writer"}


def test_multi_revise_loop_fixes_then_drops(transcript):
    t, gold = transcript
    bad = copy.deepcopy(GOOD_NOTE)
    bad["subjective"].append({"text": "Cough for nine days", "evidence": [3]})
    bad["plan"].append({"text": "Start amoxicillin", "evidence": [8]})
    rev = {
        "revised": [
            {
                "section": "subjective",
                "index": 3,
                "action": "rewrite",
                "text": "Cough for about five days",
                "evidence": [3],
            }
        ]
    }
    llm = ScriptedLLM(scripted_responder(note=bad, speaker=gold, revision=rev))
    r = run_multi(llm, t, max_rounds=2)
    texts = [c.text for _, c in r.note.claims()]
    assert "Cough for about five days" in texts
    assert "Start amoxicillin" not in texts  # still ungrounded after revisions -> dropped
    assert r.rejected_per_round[0] == 2 and r.rejected_per_round[-1] == 1
    assert r.rounds == 2 and r.dropped == 1


def test_multi_stops_revising_when_step_budget_runs_out(transcript):
    t, gold = transcript
    bad = copy.deepcopy(GOOD_NOTE)
    bad["plan"].append({"text": "Start amoxicillin", "evidence": [8]})
    llm = ScriptedLLM(scripted_responder(note=bad, speaker=gold))
    r = run_multi(llm, t, budget=Budget(max_steps=4))  # planner, speaker, extractor, writer
    assert r.status == "budget_exhausted"
    assert "Start amoxicillin" not in [c.text for _, c in r.note.claims()]


def test_multi_survives_injected_faults(transcript):
    t, gold = transcript
    inner = ScriptedLLM(scripted_responder(speaker=gold))
    faulty = FaultInjectingLLM(inner, rate=0.5, seed=3)
    r = run_multi(faulty, t, budget=Budget(max_steps=20))
    assert faulty.injected, "seed should inject at least one fault"
    assert r.status in ("ok", "failed")
    if r.status == "ok":
        assert verify(t, r.note).rejected == []
    events = {e.event for e in r.trace.events}
    assert events & {"timeout", "parse_error"}


def test_writer_failure_reports_failed(transcript):
    t, _ = transcript

    def respond(system, user):
        if "SOAP" in user and "Plan the documentation" not in user:
            raise TimeoutError("writer down")
        return scripted_responder()(system, user)

    r = run_multi(ScriptedLLM(respond), t)
    assert r.status == "failed" and r.note.claims() == []


def test_single_and_single_verify(transcript):
    t, _ = transcript
    bad = copy.deepcopy(GOOD_NOTE)
    bad["plan"].append({"text": "Start amoxicillin", "evidence": [8]})
    r1 = run_single(ScriptedLLM(scripted_responder(note=bad)), t)
    r2 = run_single(ScriptedLLM(scripted_responder(note=bad)), t, verify=True)
    assert len(r1.note.claims()) == 6 and len(r2.note.claims()) == 5 and r2.dropped == 1
    assert r1.summary()["llm_calls"] == 1


def test_agent_failed_after_retries():
    llm = ScriptedLLM(lambda s, u: "{")
    ctx = Context(llm=llm, budget=Budget(), trace=Trace(), parse_retries=1)
    with pytest.raises(AgentFailed):
        call_typed(ctx, "writer", "s", "u", SoapNote)
