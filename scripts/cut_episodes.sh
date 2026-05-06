#!/usr/bin/env bash
set -euo pipefail

MAX_JOBS="${MAX_JOBS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 <video.mp4> [more videos...]"
  echo "Set MAX_JOBS=8 to change parallelism."
  exit 2
fi

run_one() {
  "$PYTHON_BIN" katacr/dataset_builder/cut_episodes.py --path-video "$1"
}

export -f run_one
export PYTHON_BIN

printf '%s\n' "$@" | xargs -n 1 -P "$MAX_JOBS" bash -c 'run_one "$0"'
