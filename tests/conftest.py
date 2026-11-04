import json
from pathlib import Path

import pytest

from medscribe.data.primock import load_consultation
from medscribe.eval.run_notes import to_transcript

FIXTURE = Path(__file__).parent / "fixtures" / "mini"
CID = "day1_consultation01"

# Utterance ids in the fixture (merged by start time):
# 0 D hello | 1 P cough | 2 D how long | 3 P five days | 4 D fever/SOB? | 5 P no fever, SOB
# 6 D medication? | 7 P salbutamol two puffs | 8 D viral chest infection + plan


@pytest.fixture
def consultation():
    return load_consultation(FIXTURE, CID)


@pytest.fixture
def transcript(consultation):
    return to_transcript(consultation)


GOOD_NOTE = {
    "subjective": [
        {"text": "Cough for five days", "evidence": [1, 2, 3]},
        {"text": "No fever, mild shortness of breath", "evidence": [4, 5]},
        {"text": "Uses salbutamol inhaler, two puffs as needed", "evidence": [6, 7]},
    ],
    "objective": [],
    "assessment": [{"text": "Viral chest infection", "evidence": [8]}],
    "plan": [
        {"text": "Rest, fluids and paracetamol; return if not better in a week", "evidence": [8]}
    ],
}


def scripted_responder(note=GOOD_NOTE, speaker=None, revision=None):
    """Answer each agent based on the task text in its prompt."""

    def respond(system: str, user: str) -> str:
        if "Plan the documentation work" in user:
            return json.dumps(
                {"steps": ["speaker", "extractor", "writer"], "presenting_complaint": "cough"}
            )
        if "Decide which one is the DOCTOR" in user:
            return json.dumps(speaker or {"SPEAKER_A": "DOCTOR", "SPEAKER_B": "PATIENT"})
        if "List the clinically relevant facts" in user:
            return json.dumps({"entities": [{"type": "symptom", "text": "cough", "evidence": [1]}]})
        if "A verifier rejected" in user:
            return json.dumps(revision or {"revised": []})
        if "SOAP" in user:
            return json.dumps(note)
        raise AssertionError("unexpected prompt")

    return respond
