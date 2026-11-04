"""Text normalisation shared by the note metrics and the grounding verifier.

Deliberately simple and dependency-free so results are reproducible and the
behaviour is easy to audit. Clinical shorthand that appears in PriMock57's
clinician notes (``hx``, ``PMH``, ``3/7``) is expanded before comparison so a
generated note written in plain English is not penalised for not using it.
"""

from __future__ import annotations

import re

# Clinical shorthand -> plain words. Applied after lower-casing, token-wise.
ABBREVIATIONS: dict[str, str] = {
    "hx": "history",
    "pmh": "past medical history",
    "pmhx": "past medical history",
    "dh": "drug history",
    "dhx": "drug history",
    "sh": "social history",
    "shx": "social history",
    "fh": "family history",
    "fhx": "family history",
    "pc": "presenting complaint",
    "hpc": "history presenting complaint",
    "imp": "impression",
    "dx": "diagnosis",
    "ddx": "differential diagnosis",
    "rx": "treatment",
    "tx": "treatment",
    "sx": "symptoms",
    "c/o": "complains",
    "abx": "antibiotics",
    "nkda": "no known drug allergies",
    "otc": "over counter",
    "gp": "doctor",
    "ae": "emergency",
    "a&e": "emergency",
    "sob": "shortness breath",
    "soboe": "shortness breath exertion",
    "cp": "chest pain",
    "n&v": "nausea vomiting",
    "uti": "urinary tract infection",
    "urti": "upper respiratory tract infection",
    "bp": "blood pressure",
    "hr": "heart rate",
    "etoh": "alcohol",
    "adls": "daily activities",
    "adl": "daily activities",
    "llq": "left lower quadrant",
    "rlq": "right lower quadrant",
    "luq": "left upper quadrant",
    "ruq": "right upper quadrant",
    "pt": "patient",
    "pts": "patient",
    "yo": "year old",
    "yrs": "years",
    "wks": "weeks",
    "mg": "mg",
    "od": "once daily",
    "bd": "twice daily",
    "tds": "three times daily",
    "qds": "four times daily",
    "prn": "as needed",
    "nil": "no",
    "w/": "with",
    "w/o": "without",
    "+ve": "positive",
    "-ve": "negative",
}

# ``3/7`` = three days, ``2/52`` = two weeks, ``6/12`` = six months.
_DURATION = re.compile(r"\b(\d+)/(7|52|12)\b")
_DURATION_UNIT = {"7": "days", "52": "weeks", "12": "months"}

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "hundred": "100", "once": "1", "twice": "2",
}  # fmt: skip

STOPWORDS = frozenset(
    """a an the and or but if then so of to in on at by for with from as is are was were be been
    being am do does did has have had having it its it's this that these those there here i me my
    we our you your he him his she her they them their what which who whom when where why how all
    any both each few more most other some such no not only own same than too very can will just
    should would could may might must shall also into over under again further once about above
    below up down out off um uh er erm mm hmm mhm yeah yes okay ok oh right well like just really
    got get gets getting going go gone think know see say said tell told thing things bit lot
    sort kind quite actually basically maybe perhaps um uh patient doctor reports reported states
    stated says denies denied""".split()
)
# Negations carry clinical meaning ("no fever"); keep them as content tokens.
STOPWORDS = STOPWORDS - {"no", "not"}

_TAG = re.compile(r"</?[A-Z_]+/?>")
_NON_WORD = re.compile(r"[^a-z0-9/&+\-' ]+")


def strip_annotation_tags(text: str) -> str:
    """Remove PriMock57 transcription tags such as ``<UNSURE>``/``<UNIN/>``."""
    return re.sub(r"\s+", " ", _TAG.sub(" ", text)).strip()


def _stem(token: str) -> str:
    """Very light suffix stripping (pains->pain, vomiting->vomit, coughed->cough)."""
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def normalise_tokens(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lower-case, expand shorthand and number words, strip punctuation, stem lightly."""
    text = strip_annotation_tags(text).lower()
    text = _DURATION.sub(lambda m: f"{m.group(1)} {_DURATION_UNIT[m.group(2)]}", text)
    text = re.sub(r"(\d)x\b", r"\1 times", text)  # "x6/day" style is handled below too
    text = re.sub(r"\bx(\d+)", r"\1 times", text)
    text = _NON_WORD.sub(" ", text)
    out: list[str] = []
    for raw in text.split():
        tok = raw.strip("'-+/&")
        if not tok:
            continue
        expanded = ABBREVIATIONS.get(raw) or ABBREVIATIONS.get(tok) or tok
        for piece in re.split(r"[\s/]+", expanded):
            piece = piece.strip("'-+&")
            if not piece:
                continue
            piece = NUMBER_WORDS.get(piece, piece)
            if piece.endswith("'s"):
                piece = piece[:-2]
            if drop_stopwords and piece in STOPWORDS:
                continue
            out.append(_stem(piece))
    return out


def numbers_in(text: str) -> set[str]:
    """Numeric facts (digits or number words) mentioned in ``text``."""
    toks = normalise_tokens(text, drop_stopwords=False)
    return {t for t in toks if t.isdigit()}
