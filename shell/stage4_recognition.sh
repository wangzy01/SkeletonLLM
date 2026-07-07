#!/usr/bin/env bash
# Stage 4 - Recognition Refinement  [REQUIRED]
# Freezes DrAction; trains projector + LLM LoRA on MQA. Runs 3 epochs by default.
# MODEL = latest available checkpoint (Stage 3 if run, else Stage 2, else Stage 1).
# Annotation = MQA (tools/build_ntu_annotation.py).
set -euo pipefail
export STAGE=4
export MASTER_PORT=${MASTER_PORT:-32104}
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train.sh" "$@"
