"""Typed messages passed between agents.

Every agent consumes and produces pydantic models; the orchestrator never
passes raw strings around. LLM outputs are parsed into these models (with the
JSON schema sent to the model as a structured-output constraint), so a
malformed reply is a validation error the runtime can retry on, not a silent
corruption further down the pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Size bounds are written into the JSON schema sent to the model (Ollama turns
# it into a decoding grammar), which stops a small model from looping inside a
# list. Over-long lists that slip through are truncated, not rejected.
MAX_EVIDENCE = 8
MAX_CLAIMS_PER_SECTION = 12
MAX_ENTITIES = 30


def _bounded(n: int, description: str = "") -> dict:
    return {"description": description, "json_schema_extra": {"maxItems": n}}


class _Truncating(BaseModel):
    """Truncates list fields to the ``maxItems`` declared in their schema."""

    @field_validator("*", mode="before")
    @classmethod
    def _truncate(cls, v, info):
        extra = cls.model_fields[info.field_name].json_schema_extra
        limit = extra.get("maxItems") if isinstance(extra, dict) else None
        if limit is not None and isinstance(v, list):
            return v[:limit]
        return v


Role = Literal["DOCTOR", "PATIENT"]
Section = Literal["subjective", "objective", "assessment", "plan"]
SECTIONS: tuple[Section, ...] = ("subjective", "objective", "assessment", "plan")


class Turn(BaseModel):
    uid: int
    speaker: str  # SPEAKER_A / SPEAKER_B before labelling, DOCTOR / PATIENT after
    text: str


class Transcript(BaseModel):
    cid: str
    turns: list[Turn]

    def render(self, max_chars: int | None = None) -> str:
        lines = [f"[{t.uid}] {t.speaker}: {t.text}" for t in self.turns]
        out = "\n".join(lines)
        return out if max_chars is None else out[:max_chars]

    def by_id(self) -> dict[int, Turn]:
        return {t.uid: t for t in self.turns}


# --- planner --------------------------------------------------------------
AgentName = Literal["speaker", "extractor", "writer"]


class Plan(BaseModel):
    steps: list[AgentName] = Field(description="Specialists to run, in order.")
    presenting_complaint: str = Field(description="One short phrase: why the patient came in.")


# --- speaker labelling ----------------------------------------------------
class SpeakerMap(BaseModel):
    """Maps anonymous diarization clusters to clinical roles."""

    SPEAKER_A: Role
    SPEAKER_B: Role


# --- extraction -----------------------------------------------------------
EntityType = Literal[
    "symptom", "medication", "condition", "allergy", "history", "social", "finding", "plan"
]


class Entity(_Truncating):
    type: EntityType
    text: str = Field(description="Short phrase as stated in the conversation.")
    evidence: list[int] = Field(**_bounded(MAX_EVIDENCE, "Utterance ids that state this."))
    negated: bool = Field(default=False, description="True if explicitly denied, e.g. 'no fever'.")


class Entities(_Truncating):
    entities: list[Entity] = Field(**_bounded(MAX_ENTITIES))


# --- note -----------------------------------------------------------------
class Claim(_Truncating):
    text: str = Field(description="One short factual statement for the note.")
    evidence: list[int] = Field(
        **_bounded(MAX_EVIDENCE, "Utterance ids that support the statement.")
    )


class SoapNote(_Truncating):
    subjective: list[Claim] = Field(default_factory=list, **_bounded(MAX_CLAIMS_PER_SECTION))
    objective: list[Claim] = Field(default_factory=list, **_bounded(MAX_CLAIMS_PER_SECTION))
    assessment: list[Claim] = Field(default_factory=list, **_bounded(MAX_CLAIMS_PER_SECTION))
    plan: list[Claim] = Field(default_factory=list, **_bounded(MAX_CLAIMS_PER_SECTION))

    def claims(self) -> list[tuple[Section, Claim]]:
        return [(s, c) for s in SECTIONS for c in getattr(self, s)]

    def render(self) -> str:
        parts = []
        for s in SECTIONS:
            items = getattr(self, s)
            if items:
                parts.append(s.upper() + ":\n" + "\n".join(f"- {c.text}" for c in items))
        return "\n\n".join(parts)


# --- verification ---------------------------------------------------------
class Verdict(BaseModel):
    section: Section
    index: int
    claim: str
    grounded: bool
    support: float  # fraction of claim content tokens found in the cited turns
    reason: str
    recited: bool = False  # True if the verifier replaced wrong citations
    evidence: list[int] = Field(default_factory=list)  # evidence after re-citation


class VerifierReport(BaseModel):
    verdicts: list[Verdict]

    @property
    def rejected(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.grounded]


class Revision(BaseModel):
    """Writer's answer to verifier feedback: replacements for rejected claims."""

    revised: list[RevisedClaim]


class RevisedClaim(BaseModel):
    section: Section
    index: int
    action: Literal["rewrite", "drop"]
    text: str = ""
    evidence: list[int] = Field(default_factory=list)


Revision.model_rebuild()
