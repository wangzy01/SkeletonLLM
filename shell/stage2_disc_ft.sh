#!/usr/bin/env bash
# Stage 2 - Discriminative Finetuning (Disc-FT)  [OPTIONAL]
# Trains DrAction + projector on binary YES/NO hard-negative judgments.
# MODEL = Stage 1 checkpoint. Annotation = tools/build_discft_annotation.py.
set -euo pipefail
export STAGE=2
export MASTER_PORT=${MASTER_PORT:-32102}
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train.sh" "$@"
