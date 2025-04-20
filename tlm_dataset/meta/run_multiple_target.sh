#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <target-substring>"
  exit 1
fi

TARGET="$1"

# 1. 拿到所有 GPU 的名称（不带表头）
# GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)

# # 2. 判断 TARGET 是否为任意一行 GPU 名称的子串
# if echo "$GPU_NAMES" | grep -Fqi "$TARGET"; then
#   echo "✅ 目标 '$TARGET' 在以下 GPU 名称中找到："
#   echo "$GPU_NAMES" | grep -Fi "$TARGET"
# else
#   echo "❌ 错误：目标 '$TARGET' 未出现在任何 GPU 名称里。当前可用 GPU："
#   echo "$GPU_NAMES"
#   exit 2
# fi

# 3. 如果匹配，通过 run.py 传给 --target
PYTHONUNBUFFERED=1 python run.py --target="$TARGET" --for_type=for_finetuning --finetuning_init=True |& tee run_$TARGET.log

#tmux new-session -d -s $TARGET "PYTHONUNBUFFERED=1 python run.py --target="nvidia/nvidia-$TARGET" --for_type=for_finetuning --finetuning_init=True |& tee run_$TARGET.log"
