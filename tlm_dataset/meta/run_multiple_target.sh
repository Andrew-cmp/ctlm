#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <target-substring>"
  exit 1
fi

TARGET="$1"
model="${TARGET#*-}" 


# 3. 如果匹配，通过 run.py 传给 --target
PYTHONUNBUFFERED=1 python run.py --target="$TARGET" --for_type=for_finetuning --finetuning_init=False |& tee run_$model.log

#tmux new-session -d -s $TARGET "PYTHONUNBUFFERED=1 python run.py --target="nvidia/nvidia-$TARGET" --for_type=for_finetuning --finetuning_init=True |& tee run_$TARGET.log"
