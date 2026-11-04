#!/usr/bin/env bash
# Fetch PriMock57 (CC BY 4.0, Babylon Health) into data/primock57 at a pinned commit.
# Audio (~1.7 GB, 114 wav files) is stored with git-lfs; pass --no-audio to skip it
# (the note-generation experiments only need transcripts + notes).
set -euo pipefail
COMMIT=cd2ac707ad03cb4d2531f4ec6b90c659bf4357c5
DEST=${DEST:-data/primock57}
mkdir -p "$(dirname "$DEST")"
if [ ! -d "$DEST/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/babylonhealth/primock57.git "$DEST"
fi
git -C "$DEST" checkout -q "$COMMIT"
if [ "${1:-}" != "--no-audio" ]; then
  git -C "$DEST" lfs pull
fi
echo "PriMock57 at $(git -C "$DEST" rev-parse --short HEAD) in $DEST"
