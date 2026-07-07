#!/usr/bin/env bash
# Stage 1 - Render Warm-up  [REQUIRED]
# Trains DrAction + projector (LLM and vision backbone frozen).
# MODEL = base InternVL3-8B. Annotation = MQA (tools/build_ntu_annotation.py).
set -euo pipefail
export STAGE=1
export MASTER_PORT=${MASTER_PORT:-32101}
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train.sh" "$@"
