#!/bin/bash
set -euo pipefail

NUM_PAIRS_LIST=(100)
OVER_CLUSTER_FACTOR=2

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

for NUM_PAIRS in "${NUM_PAIRS_LIST[@]}"; do
  OUT_DIR="outputs/flickr_num${NUM_PAIRS}_ocf${OVER_CLUSTER_FACTOR}"
  mkdir -p "$OUT_DIR"

  echo
  echo "=============================================================="
  echo "[$(timestamp)] START  num_pairs=${NUM_PAIRS}, over_cluster_factor=${OVER_CLUSTER_FACTOR}"
  echo "=============================================================="

  echo "[$(timestamp)] Step 1: 运行 distill..."
  HF_ENDPOINT=https://hf-mirror.com python pds_distill.py \
    --mode distill \
    --dataset flickr \
    --data_root './data/datasets/Flickr30k' \
    --num_pairs "$NUM_PAIRS" \
    --over_cluster_factor "$OVER_CLUSTER_FACTOR" \
    --rm_ratio 0.3 \
    --joint_alpha 0.6


  echo "[$(timestamp)] Step 2: 运行 eval..."
  {
    echo "=============================================================="
    echo "[$(timestamp)] EVAL START num_pairs=${NUM_PAIRS}, over_cluster_factor=${OVER_CLUSTER_FACTOR}"
    echo "=============================================================="

    HF_ENDPOINT=https://hf-mirror.com python pds_distill.py \
      --mode eval \
      --dataset flickr \
      --data_root './data/datasets/Flickr30k' \
      --num_pairs "$NUM_PAIRS" \
      --epoch_eval_train 200 \
      --noise_std 0.1 \
      --over_cluster_factor "$OVER_CLUSTER_FACTOR"

    echo "=============================================================="
    echo "[$(timestamp)] EVAL END   num_pairs=${NUM_PAIRS}, over_cluster_factor=${OVER_CLUSTER_FACTOR}"
    echo "=============================================================="
  } 2>&1 | tee "$OUT_DIR/xxxeval.log"

  echo "[$(timestamp)] ✅ 完成: num_pairs=${NUM_PAIRS}, over_cluster_factor=${OVER_CLUSTER_FACTOR}"
  echo "[$(timestamp)] 日志位置: $OUT_DIR/eval.log"
done

echo
echo "[$(timestamp)] 🎉 全部跑完！"
