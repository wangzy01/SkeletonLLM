#!/usr/bin/env bash
# Stage 3 - Causal Reasoning Distillation (CR-Distill)  [OPTIONAL]
# Trains DrAction + projector + LLM LoRA to reproduce teacher causal rationales.
# MODEL = Stage 2 checkpoint. Annotation = tools/build_crdistill_annotation.py.
set -euo pipefail
export STAGE=3
export MASTER_PORT=${MASTER_PORT:-32103}
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train.sh" "$@"
