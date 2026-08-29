#!/usr/bin/env bash
# Anotasi 3 model SEQUENTIAL di satu A100 40GB, jalur IN-PROCESS (tanpa server).
# Tiap model dapat penuh 40GB -> KV cache maksimal -> throughput tertinggi.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

CHUNK=${CHUNK:-25}
MML=${MML:-8192}          # 8192 muat chunk 25-50; ukur ulang dgn --check-tokens bila diubah
WAVE=${WAVE:-1000}

# tag|model_id   — WAJIB tiga keluarga pretraining berbeda
# ctx: Qwen2.5 32k, Mistral-v0.2 32k, Yi-1.5-16K 16k.
# JANGAN pakai 01-ai/Yi-1.5-9B-Chat biasa: konteksnya hanya 4.096, tidak muat chunk-25.
MODELS=(
  "qwen|Qwen/Qwen2.5-7B-Instruct"
  "mistral|mistralai/Mistral-7B-Instruct-v0.2"
  "yi|01-ai/Yi-1.5-9B-Chat-16K"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r TAG MID <<< "$entry"
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] $TAG  <-  $MID"

  python3 annotate_offline.py --tag "$TAG" --model "$MID" \
      --chunk "$CHUNK" --max-model-len "$MML" --wave "$WAVE" \
    || { echo "  !! $TAG GAGAL — lanjut ke model berikutnya"; continue; }

  echo "[$(date +%H:%M:%S)] $TAG selesai -> $(wc -l < "ann_${TAG}.jsonl" 2>/dev/null || echo 0) baris"
done

echo "=============================================================="
python3 agreement.py ann_qwen.jsonl ann_mistral.jsonl ann_yi.jsonl | tee agreement_report.txt
