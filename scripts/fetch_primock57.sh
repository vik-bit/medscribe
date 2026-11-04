#!/usr/bin/env bash
# Fetch PriMock57 (CC BY 4.0, Babylon Health) into data/primock57 at a pinned commit.
# Audio (~1.7 GB, 114 wav files) is stored with git-lfs. If git-lfs is not available
# (e.g. a fresh Colab runtime) each file is downloaded from GitHub's LFS media URL
# instead and checked against the sha256 in its pointer file.
# Pass --no-audio to skip audio (the note-generation experiments only need transcripts + notes).
set -euo pipefail
COMMIT=cd2ac707ad03cb4d2531f4ec6b90c659bf4357c5
REPO=babylonhealth/primock57
DEST=${DEST:-data/primock57}
mkdir -p "$(dirname "$DEST")"
if [ ! -d "$DEST/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone "https://github.com/$REPO.git" "$DEST"
fi
git -C "$DEST" -c advice.detachedHead=false checkout -q "$COMMIT"

if [ "${1:-}" = "--no-audio" ]; then
  echo "PriMock57 at $(git -C "$DEST" rev-parse --short HEAD) in $DEST (no audio)"
  exit 0
fi

is_pointer() { head -c 40 "$1" | grep -q "git-lfs.github.com/spec"; }

if git lfs version >/dev/null 2>&1; then
  git -C "$DEST" lfs install --local >/dev/null
  git -C "$DEST" lfs pull || echo "git lfs pull failed; falling back to direct download" >&2
else
  echo "git-lfs not installed; downloading audio directly" >&2
fi

n=0
for f in "$DEST"/audio/*.wav; do
  is_pointer "$f" || continue
  oid=$(sed -n 's/^oid sha256:\([0-9a-f]*\)$/\1/p' "$f")
  name=$(basename "$f")
  curl -fsSL --retry 3 -o "$f.part" "https://media.githubusercontent.com/media/$REPO/$COMMIT/audio/$name"
  got=$(sha256sum "$f.part" | cut -d' ' -f1)
  if [ "$got" != "$oid" ]; then
    echo "checksum mismatch for $name" >&2
    rm -f "$f.part"
    exit 1
  fi
  mv "$f.part" "$f"
  n=$((n + 1))
done
[ "$n" -gt 0 ] && echo "downloaded $n audio files directly"

bad=0
for f in "$DEST"/audio/*.wav; do
  if is_pointer "$f" || [ "$(head -c 4 "$f")" != "RIFF" ]; then
    echo "not a wav file: $f" >&2
    bad=$((bad + 1))
  fi
done
if [ "$bad" -gt 0 ]; then
  echo "$bad audio files are missing or still git-lfs pointers" >&2
  exit 1
fi
echo "PriMock57 at $(git -C "$DEST" rev-parse --short HEAD) in $DEST ($(ls "$DEST"/audio/*.wav | wc -l) wav files OK)"
