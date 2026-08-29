#!/usr/bin/env bash
# Anotasi 3 model SEQUENTIAL di satu A100 40GB.
# Tiap model dapat penuh 40GB -> KV cache maksimal -> throughput tertinggi.
# Sesuaikan MODELS dengan model yang benar-benar ada di server.
set -uo pipefail

PORT=8000
BASE="http://localhost:${PORT}/v1"
CHUNK=${CHUNK:-25}
CONC=${CONC:-64}

# tag:huggingface_id — WAJIB tiga keluarga pretraining berbeda
MODELS=(
  "qwen:QuantTrio/Qwen3.6-35B-A3B-AWQ"
  "gemma:cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4"
  "nemotron:nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
)

wait_ready() {           # tunggu server siap, maks 15 menit
  for _ in $(seq 1 180); do
    curl -sf "${BASE}/models" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

for entry in "${MODELS[@]}"; do
  TAG="${entry%%:*}"; MID="${entry#*:}"
  OUT="ann_${TAG}.jsonl"

  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] $TAG  <-  $MID"

  vllm serve "$MID" \
      --port "$PORT" \
      --max-num-batched-tokens 16384 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.92 \
      --moe-backend marlin \
      --enable-prefix-caching \
      > "vllm_${TAG}.log" 2>&1 &
  VPID=$!

  if ! wait_ready; then
    echo "  !! server $TAG gagal siap — lihat vllm_${TAG}.log"
    kill $VPID 2>/dev/null; wait $VPID 2>/dev/null; continue
  fi
  echo "[$(date +%H:%M:%S)] server siap"

  # probe dulu: kalau 'output invalid' > 5%, turunkan CHUNK sebelum produksi
  python3 probe.py --base-url "$BASE" --model "$MID" --chunk "$CHUNK" \
          --concurrency 16 --calls 16 2>&1 | tee "probe_${TAG}.txt"

  python3 annotate.py --tag "$TAG" --base-url "$BASE" --model "$MID" \
          --chunk "$CHUNK" --concurrency "$CONC" --out "$OUT"

  kill $VPID 2>/dev/null; wait $VPID 2>/dev/null
  echo "[$(date +%H:%M:%S)] $TAG selesai -> $(wc -l < "$OUT") baris"
done

echo "=============================================================="
python3 agreement.py ann_qwen.jsonl ann_gemma.jsonl ann_nemotron.jsonl \
        | tee agreement_report.txt
