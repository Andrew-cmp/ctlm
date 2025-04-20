#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <target-substring>"
  exit 1
fi

TARGET="$1"

# 1. 拿到所有 GPU 的名称（不带表头）
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)
model="${TARGET#*-}" 
# 2. 判断 TARGET 是否为任意一行 GPU 名称的子串
if echo "$GPU_NAMES" | grep -Fqi "$model"; then
  echo "✅ 目标 '$model' 在以下 GPU 名称中找到："
  echo "$GPU_NAMES" | grep -Fi "$model"
else
  echo "❌ 错误：目标 '$model' 未出现在任何 GPU 名称里。当前可用 GPU："
  echo "$GPU_NAMES"
  exit 2
fi

# 3. 如果匹配，通过 run.py 传给 --target
PYTHONUNBUFFERED=1 python run_measure.py --reg_times=-1 --cuda_id1=0 --cuda_id2=1 --target=$TARGET |& tee run.log

