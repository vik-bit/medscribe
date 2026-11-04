"""Planner, specialist agents and the verifier.

Prompt layout: every LLM agent sends the same system prompt and starts its
user message with the same transcript block, then appends its own task. With a
local Ollama server that shared prefix stays in the KV cache between
sequential agent calls, so later agents only pay prefill for their task text.
Token counts are still the full prompt length (Ollama reports it that way), so
the cost comparison with the single-agent baseline is not flattered.
"""

from __future__ import annotations

import json

from medscribe.agents.messages import (
    SECTIONS,
    Claim,
    Entities,
    Plan,
    Revision,
    SoapNote,
    SpeakerMap,
    Transcript,
    Verdict,
    VerifierReport,
)
from medscribe.agents.runtime import Context, call_typed
from medscribe.grounding import best_window_support, support, unsupported_numbers

SYSTEM = (
    "You are part of a clinical documentation system working on a recorded "
    "doctor-patient consultation. Each utterance is prefixed with its id in square "
    "brackets. Use only information stated in the conversation; never add facts, "
    "doses, diagnoses or advice that were not said. Reply with JSON only."
)


def transcript_block(t: Transcript) -> str:
    return f"TRANSCRIPT ({t.cid}):\n{t.render()}\n\nTASK:\n"


# --- planner --------------------------------------------------------------
DEFAULT_PLAN = Plan(steps=["speaker", "extractor", "writer"], presenting_complaint="")


def plan(ctx: Context, t: Transcript) -> Plan:
    """Pick specialists and summarise the presenting complaint.

    The planner's output is validated: unknown steps are dropped, duplicates
    removed, and the writer is always last. If the call fails the default plan
    is used (and the fallback is recorded in the trace).
    """
    task = (
        "Plan the documentation work. Available specialists: 'speaker' (decide which "
        "speaker is the doctor), 'extractor' (list symptoms, medications, history, "
        "findings and plan items with utterance ids), 'writer' (write the SOAP note). "
        "Return the steps to run in order and a short presenting complaint."
    )
    try:
        p = call_typed(ctx, "planner", SYSTEM, transcript_block(t) + task, Plan)
    except Exception as e:  # noqa: BLE001
        ctx.trace.add("planner", "info", detail=f"fallback to default plan ({type(e).__name__})")
        return DEFAULT_PLAN
    steps = [s for i, s in enumerate(p.steps) if s != "writer" and s not in p.steps[:i]]
    return Plan(steps=[*steps, "writer"], presenting_complaint=p.presenting_complaint[:200])


# --- speaker labelling ----------------------------------------------------
def heuristic_speaker_map(t: Transcript) -> SpeakerMap:
    """Fallback: the cluster that asks more questions is the doctor."""
    q = {"SPEAKER_A": 0, "SPEAKER_B": 0}
    for turn in t.turns:
        if turn.speaker in q:
            q[turn.speaker] += turn.text.count("?")
    doc_a = q["SPEAKER_A"] >= q["SPEAKER_B"]
    return SpeakerMap(
        SPEAKER_A="DOCTOR" if doc_a else "PATIENT", SPEAKER_B="PATIENT" if doc_a else "DOCTOR"
    )


def label_speakers(ctx: Context, t: Transcript) -> tuple[SpeakerMap, str]:
    task = (
        "The speakers were separated by a diarization system into SPEAKER_A and "
        "SPEAKER_B. Decide which one is the DOCTOR and which is the PATIENT. "
        "They must be different. The doctor greets the patient, asks most of the questions, "
        "and gives the diagnosis and advice; the patient describes their symptoms."
    )
    try:
        m = call_typed(ctx, "speaker", SYSTEM, transcript_block(t) + task, SpeakerMap)
        if m.SPEAKER_A != m.SPEAKER_B:
            return m, "llm"
        ctx.trace.add("speaker", "info", detail="both clusters given same role; heuristic used")
    except Exception as e:  # noqa: BLE001
        ctx.trace.add("speaker", "info", detail=f"heuristic fallback ({type(e).__name__})")
    return heuristic_speaker_map(t), "heuristic"


def relabel(t: Transcript, m: SpeakerMap) -> Transcript:
    mapping = {"SPEAKER_A": m.SPEAKER_A, "SPEAKER_B": m.SPEAKER_B}
    return Transcript(
        cid=t.cid,
        turns=[
            tt.model_copy(update={"speaker": mapping.get(tt.speaker, tt.speaker)}) for tt in t.turns
        ],
    )


# --- extraction -----------------------------------------------------------
def extract(ctx: Context, t: Transcript) -> Entities:
    task = (
        "List the clinically relevant facts: symptoms (with duration/character), "
        "medications (with dose if stated), conditions, allergies, past/family "
        "history, social history, examination findings and agreed plan items. "
        "Set negated=true for things the patient denies (e.g. 'no blood in stool'). "
        "For each, give the ids of the utterances where it is said, including the "
        "doctor's question if the patient's answer is just 'yes' or 'no'."
    )
    try:
        return call_typed(ctx, "extractor", SYSTEM, transcript_block(t) + task, Entities)
    except Exception as e:  # noqa: BLE001
        ctx.trace.add(
            "extractor", "info", detail=f"continuing without entities ({type(e).__name__})"
        )
        return Entities(entities=[])


# --- writer ---------------------------------------------------------------
NOTE_TASK = (
    "Write the consultation note in SOAP format as short factual claims, typically "
    "10-20 claims in total, one fact per claim. "
    "subjective: presenting complaint, history, symptoms (including relevant "
    "negatives), medications, allergies, social and family history. objective: "
    "examination findings or observations that were actually stated (may be empty). "
    "assessment: the doctor's stated impression or diagnosis. plan: what the doctor "
    "said would happen next (treatment, advice, safety-netting, follow-up). "
    "Each claim is one plain-English sentence stating a single fact (never a list of "
    "ids, never just a heading). Put the ids of the utterances that support it in "
    "'evidence'; cite both the question and the answer when the answer alone is "
    "'yes' or 'no'. Example of the format (content is illustrative only):\n"
    '{"subjective": [{"text": "Cough for five days, worse at night.", "evidence": [3, 4]}, '
    '{"text": "No fever.", "evidence": [9, 10]}], "objective": [], '
    '"assessment": [{"text": "Likely viral chest infection.", "evidence": [41]}], '
    '"plan": [{"text": "Rest and fluids; return if not better in a week.", "evidence": [43, 45]}]}'
)


def write_note(ctx: Context, t: Transcript, entities: Entities, p: Plan) -> SoapNote:
    extra = ""
    if p.presenting_complaint:
        extra += f"\nPresenting complaint (from planner): {p.presenting_complaint}"
    if entities.entities:
        ents = [e.model_dump() for e in entities.entities]
        extra += f"\nExtracted facts to cover (verify against the transcript):\n{json.dumps(ents)}"
    return call_typed(ctx, "writer", SYSTEM, transcript_block(t) + NOTE_TASK + extra, SoapNote)


def revise(ctx: Context, t: Transcript, note: SoapNote, report: VerifierReport) -> SoapNote:
    """Ask the writer to rewrite (with correct citations) or drop rejected claims."""
    rejected = [
        {"section": v.section, "index": v.index, "claim": v.claim, "problem": v.reason}
        for v in report.rejected
    ]
    task = (
        "A verifier rejected these note claims because the cited utterances do not "
        "support them. For each one either rewrite it so it only states what the "
        "conversation says and cite the right utterance ids, or drop it if the "
        f"conversation does not support it.\nREJECTED: {json.dumps(rejected)}"
    )
    rev = call_typed(ctx, "writer", SYSTEM, transcript_block(t) + task, Revision)
    return apply_revision(note, rev)


def apply_revision(note: SoapNote, rev: Revision) -> SoapNote:
    sections = {s: list(getattr(note, s)) for s in SECTIONS}
    drop: set[tuple[str, int]] = set()
    for r in rev.revised:
        items = sections.get(r.section)
        if items is None or not 0 <= r.index < len(items):
            continue
        if r.action == "drop" or not r.text.strip():
            drop.add((r.section, r.index))
        else:
            items[r.index] = Claim(text=r.text, evidence=r.evidence)
    return SoapNote(
        **{
            s: [c for i, c in enumerate(items) if (s, i) not in drop]
            for s, items in sections.items()
        }
    )


# --- verifier (deterministic) ---------------------------------------------
SUPPORT_THRESHOLD = 0.5
RECITE_WINDOW = 4


def _check(claim: str, texts: list[str], threshold: float) -> tuple[float, str]:
    s = support(claim, texts)
    bad_nums = unsupported_numbers(claim, texts)
    if bad_nums:
        return s, f"numbers {sorted(bad_nums)} not in cited utterances"
    if s < threshold:
        return s, f"only {s:.0%} of the claim's words appear in the cited utterances"
    return s, "ok"


def verify(
    t: Transcript,
    note: SoapNote,
    threshold: float = SUPPORT_THRESHOLD,
    *,
    recite: bool = True,
    window: int = RECITE_WINDOW,
) -> VerifierReport:
    """Reject every claim that is not grounded in a transcript span.

    A claim is grounded in a set of utterances iff at least ``threshold`` of
    its content tokens appear in them and every number it states appears in
    them. The claim's own citations are checked first. If they fail and
    ``recite`` is on, the verifier searches for the best window of ``window``
    consecutive utterances and, if that window passes the same test, replaces
    the citations (small models often state a supported fact but cite the
    wrong ids). Anything still unsupported is rejected.

    Deliberately not an LLM: the component that guards against hallucination
    should not be able to hallucinate itself.
    """
    by_id = t.by_id()
    texts = [x.text for x in t.turns]
    verdicts = []
    for section in SECTIONS:
        for i, c in enumerate(getattr(note, section)):
            cited = [by_id[e].text for e in c.evidence if e in by_id]
            if cited:
                s, reason = _check(c.text, cited, threshold)
            else:
                s, reason = 0.0, "no valid utterance id cited"
            v = Verdict(section=section, index=i, claim=c.text, grounded=reason == "ok",
                        support=round(s, 3), reason=reason, evidence=list(c.evidence))  # fmt: skip
            if not v.grounded and recite:
                _, start = best_window_support(c.text, texts, window)
                span = t.turns[start : start + window]
                s2, reason2 = _check(c.text, [x.text for x in span], threshold)
                if reason2 == "ok":
                    v = Verdict(section=section, index=i, claim=c.text, grounded=True,
                                support=round(s2, 3), reason=f"re-cited ({reason})", recited=True,
                                evidence=[x.uid for x in span])  # fmt: skip
            verdicts.append(v)
    return VerifierReport(verdicts=verdicts)


def apply_verdicts(note: SoapNote, report: VerifierReport) -> SoapNote:
    """Keep grounded claims (with re-cited evidence where the verifier fixed it)."""
    keep = {(v.section, v.index): v for v in report.verdicts if v.grounded}
    return SoapNote(
        **{
            s: [
                Claim(text=c.text, evidence=keep[(s, i)].evidence)
                for i, c in enumerate(getattr(note, s))
                if (s, i) in keep
            ]
            for s in SECTIONS
        }
    )


def drop_rejected(note: SoapNote, report: VerifierReport) -> SoapNote:
    """Alias kept for callers: drop rejected claims, apply re-citations."""
    return apply_verdicts(note, report)
