#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
EXTRA_ARGS="${2:-}"
REPLAY_DATASET="${REPLAY_DATASET:-/data/user/zhihengwu/Coding/dataset/Clash-Royale-Replay-Dataset/golem_ai}"

CUDA_VISIBLE_DEVICES="$GPU_ID" python katacr/policy/offline_rl/train.py --wandb \
  --total-epochs 20 \
  --batch-size 32 \
  --nominal-batch-size 128 \
  --cnn-mode "cnn_blocks" \
  --name "StARformer_3L_v0.8_golem_ai_interval2" \
  --pred-card-idx \
  --random-interval 2 \
  --n-step 50 \
  --replay-dataset "$REPLAY_DATASET" \
  $EXTRA_ARGS
