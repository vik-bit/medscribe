"""Utterance-level ASR segments from PriMock57's per-channel audio + TextGrids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from medscribe.data.primock import consultation_ids, make_splits, parse_textgrid
from medscribe.text import strip_annotation_tags

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Segment:
    cid: str
    channel: str  # doctor | patient
    wav: Path
    start: float
    end: float
    text: str  # reference, annotation tags removed

    @property
    def key(self) -> str:
        return f"{self.cid}:{self.channel}:{self.start:.2f}"

    @property
    def duration(self) -> float:
        return self.end - self.start


def segments_for(
    root: Path, cids: list[str], *, min_s: float = 0.5, max_s: float = 30.0
) -> list[Segment]:
    """All usable utterances of ``cids``.

    Utterances containing ``<UNIN/>`` or ``<INAUDIBLE_SPEECH/>`` are excluded because
    their reference is incomplete; ``<UNSURE>`` tags are stripped and the text
    kept. Segments outside [min_s, max_s] are dropped (Whisper's window is 30 s).
    """
    out = []
    for cid in cids:
        for ch in ("doctor", "patient"):
            tg = (root / "transcripts" / f"{cid}_{ch}.TextGrid").read_text(encoding="utf-8")
            for iv in parse_textgrid(tg):
                if "<UNIN" in iv.text or "<INAUDIBLE" in iv.text:
                    continue
                text = strip_annotation_tags(iv.text)
                if not text or not (min_s <= iv.end - iv.start <= max_s):
                    continue
                out.append(
                    Segment(cid, ch, root / "audio" / f"{cid}_{ch}.wav", iv.start, iv.end, text)
                )
    return out


def split_segments(root: Path, split: str, **kw) -> list[Segment]:
    return segments_for(root, make_splits(consultation_ids(root))[split], **kw)


_LFS_HINT = "Run `bash scripts/fetch_primock57.sh` (it downloads the audio even without git-lfs)."


def check_audio(segs: list[Segment]) -> None:
    """Fail fast, before any model download, if the audio is missing or is git-lfs pointers."""
    for wav in sorted({s.wav for s in segs}):
        if not wav.exists():
            raise FileNotFoundError(f"{wav} is missing. {_LFS_HINT}")
        with open(wav, "rb") as f:
            head = f.read(64)
        if b"git-lfs.github.com/spec" in head:
            raise RuntimeError(f"{wav} is a git-lfs pointer, not audio. {_LFS_HINT}")
        if not head.startswith(b"RIFF"):
            raise RuntimeError(f"{wav} is not a RIFF/WAV file. {_LFS_HINT}")


def load_audio(seg: Segment):
    """Float32 mono samples of the segment (files are 16 kHz mono PCM)."""
    import soundfile as sf

    with sf.SoundFile(seg.wav) as f:
        if f.samplerate != SAMPLE_RATE:
            raise ValueError(f"{seg.wav}: expected 16 kHz, got {f.samplerate}")
        f.seek(int(seg.start * SAMPLE_RATE))
        return f.read(int(seg.duration * SAMPLE_RATE), dtype="float32", always_2d=False)
