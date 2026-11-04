"""Loader for PriMock57 (Korfiatis et al., 2022; CC BY 4.0).

Layout of https://github.com/babylonhealth/primock57 (cloned by
``scripts/fetch_primock57.sh`` into ``data/primock57``)::

    audio/day1_consultation01_doctor.wav     # one channel per speaker (git-lfs)
    transcripts/day1_consultation01_doctor.TextGrid
    notes/day1_consultation01.json           # clinician-written note + highlights

The doctor and patient are recorded on separate channels, so the per-channel
TextGrids give ground-truth speaker turns. We expose them two ways:

* ``Consultation.utterances`` with the true role (``DOCTOR``/``PATIENT``);
* ``Consultation.diarized()`` which replaces roles by anonymous cluster ids
  (``SPEAKER_A``/``SPEAKER_B``) the way a diarization system would, so the
  speaker-labelling agent has a real job to do.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from medscribe.text import strip_annotation_tags

_INTERVAL = re.compile(
    r'xmin = ([\d.eE+-]+)\s*\n\s*xmax = ([\d.eE+-]+)\s*\n\s*text = "((?:[^"]|"")*)"',
    re.MULTILINE,
)
ROLES = ("DOCTOR", "PATIENT")


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    text: str  # raw text, annotation tags kept


def parse_textgrid(text: str) -> list[Interval]:
    """Parse the (long-format) Praat TextGrid intervals that have non-empty text."""
    out = []
    for m in _INTERVAL.finditer(text):
        raw = m.group(3).replace('""', '"').strip()
        if raw:
            out.append(Interval(float(m.group(1)), float(m.group(2)), raw))
    return out


@dataclass(frozen=True)
class Utterance:
    uid: int
    start: float
    end: float
    speaker: str  # DOCTOR / PATIENT, or SPEAKER_A / SPEAKER_B once diarized
    text: str  # annotation tags removed

    def render(self) -> str:
        return f"[{self.uid}] {self.speaker}: {self.text}"


@dataclass
class Consultation:
    cid: str  # e.g. "day1_consultation01"
    utterances: list[Utterance]
    note: str
    highlights: list[str] = field(default_factory=list)
    presenting_complaint: str = ""

    def diarized(self, seed: int = 0) -> tuple[list[Utterance], dict[str, str]]:
        """Anonymise roles to cluster ids. Returns utterances and the gold mapping.

        Which role becomes SPEAKER_A is randomised per consultation (seeded), so
        "A is always the doctor" cannot be learned.
        """
        rng = random.Random(f"{seed}:{self.cid}")
        clusters = ["SPEAKER_A", "SPEAKER_B"]
        rng.shuffle(clusters)
        role_to_cluster = dict(zip(ROLES, clusters, strict=True))
        utts = [
            Utterance(u.uid, u.start, u.end, role_to_cluster[u.speaker], u.text)
            for u in self.utterances
        ]
        return utts, {c: r for r, c in role_to_cluster.items()}


def _merge(doctor: list[Interval], patient: list[Interval]) -> list[Utterance]:
    rows = [(iv, "DOCTOR") for iv in doctor] + [(iv, "PATIENT") for iv in patient]
    rows.sort(key=lambda r: (r[0].start, r[0].end))
    utts = []
    for iv, role in rows:
        text = strip_annotation_tags(iv.text)
        if text:
            utts.append(Utterance(len(utts), round(iv.start, 2), round(iv.end, 2), role, text))
    return utts


def consultation_ids(root: Path) -> list[str]:
    return sorted(p.stem for p in (root / "notes").glob("day*_consultation*.json"))


def load_consultation(root: Path, cid: str) -> Consultation:
    tdir = root / "transcripts"
    doctor = parse_textgrid((tdir / f"{cid}_doctor.TextGrid").read_text(encoding="utf-8"))
    patient = parse_textgrid((tdir / f"{cid}_patient.TextGrid").read_text(encoding="utf-8"))
    meta = json.loads((root / "notes" / f"{cid}.json").read_text(encoding="utf-8"))
    return Consultation(
        cid=cid,
        utterances=_merge(doctor, patient),
        note=meta["note"],
        highlights=list(meta.get("highlights", [])),
        presenting_complaint=meta.get("presenting_complaint", ""),
    )


def load_all(root: Path) -> list[Consultation]:
    return [load_consultation(root, cid) for cid in consultation_ids(root)]


# ---------------------------------------------------------------------------
# Splits. Consultation-level (never utterance-level) so no conversation leaks
# between train and test. Fixed seed; the lists are written to
# results/splits.json so anyone can check them.
# ---------------------------------------------------------------------------
SPLIT_SEED = 57


def make_splits(cids: list[str], n_dev: int = 5, n_test: int = 12) -> dict[str, list[str]]:
    ids = sorted(cids)
    random.Random(SPLIT_SEED).shuffle(ids)
    test = sorted(ids[:n_test])
    dev = sorted(ids[n_test : n_test + n_dev])
    train = sorted(ids[n_test + n_dev :])
    return {"train": train, "dev": dev, "test": test}
