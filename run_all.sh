#!/usr/bin/env bash
# Anotasi 3 model SEQUENTIAL di satu A100 40GB, jalur IN-PROCESS (tanpa server).
# Tiap model dapat penuh 40GB -> KV cache maksimal -> throughput tertinggi.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

CHUNK=${CHUNK:-25}
GU=${GU:-0.60}          # MIG 7g.40gb: usable ~24 GiB, bukan 39,39
MML=${MML:-8192}          # 8192 muat chunk 25-50; ukur ulang dgn --check-tokens bila diubah
WAVE=${WAVE:-1000}

# tag|model_id   — WAJIB tiga keluarga pretraining berbeda
# Semua AWQ INT4: MIG hanya menyediakan ~24 GiB, BF16 7B (14,2 GiB) menyisakan
# KV cache terlalu kecil. AWQ ~5 GiB -> KV ~17 GiB. Kualitas nyaris sama, memori 1/3.
# ctx: Qwen2.5 32k, Mistral-v0.2 32k, Llama-3.1 128k. Tidak ada sliding window.
# Yi-1.5 dicoret: tidak ada AWQ tepercaya, dan BF16-nya (17,8 GiB) tidak muat.
MODELS=(
  "qwen|Qwen/Qwen2.5-7B-Instruct-AWQ"
  "mistral|TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
  "llama|hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r TAG MID <<< "$entry"
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] $TAG  <-  $MID"

  python3 annotate_offline.py --tag "$TAG" --model "$MID" \
      --chunk "$CHUNK" --max-model-len "$MML" --wave "$WAVE" --gpu-util "$GU" \
    || { echo "  !! $TAG GAGAL — lanjut ke model berikutnya"; continue; }

  echo "[$(date +%H:%M:%S)] $TAG selesai -> $(wc -l < "ann_${TAG}.jsonl" 2>/dev/null || echo 0) baris"
done

echo "=============================================================="
python3 agreement.py ann_qwen.jsonl ann_mistral.jsonl ann_llama.jsonl | tee agreement_report.txt
