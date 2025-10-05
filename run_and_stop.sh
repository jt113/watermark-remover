#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="run_and_stop.log"

START_TS=$(date --iso-8601=seconds)
echo "${START_TS} - Starting deep inpaint run" | tee -a "$LOG_FILE"

python scripts/run_deep_inpaint.py \
  --frames frames \
  --overlays overlays.json \
  --output-video cleaned_deep.mp4 \
  --context 6 \
  --python "$(which python)" \
  --extra --deep-weights third_party/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth \
  --extra --deep-weights-arg=--ckpt

STATUS=$?

END_TS=$(date --iso-8601=seconds)
echo "${END_TS} - Deep inpaint completed with status ${STATUS}" | tee -a "$LOG_FILE"

echo "Halting instance via sudo shutdown -h now" | tee /tmp/stop.log
sudo shutdown -h now || true

exit "$STATUS"
