#!/usr/bin/env bash
# Anotasi 3 model SEQUENTIAL di satu A100 40GB. Target: driver 525 / vLLM 0.6.6.
# Tiap model dapat penuh 40GB -> KV cache maksimal -> throughput tertinggi.
set -uo pipefail

PORT=8000
BASE="http://localhost:${PORT}/v1"
CHUNK=${CHUNK:-25}          # JANGAN naikkan tanpa membaca catatan sliding window di bawah
CONC=${CONC:-64}

# tag|model_id|flag_khusus   — WAJIB tiga keluarga pretraining berbeda
#
# Ketiganya TIDAK memakai sliding window (Qwen2.5 use_sliding_window=False;
# Mistral-v0.2 dan Yi-1.5 tidak punya), jadi prefix caching jalan tanpa
# --disable-sliding-window. Ketiganya juga TIDAK gated di HuggingFace.
MODELS=(
  "qwen|Qwen/Qwen2.5-7B-Instruct|--enable-prefix-caching"
  "mistral|mistralai/Mistral-7B-Instruct-v0.2|--enable-prefix-caching"
  "yi|01-ai/Yi-1.5-9B-Chat|--enable-prefix-caching"
)

wait_ready() {           # tunggu server siap, maks 15 menit
  for _ in $(seq 1 180); do
    curl -sf "${BASE}/models" >/dev/null 2>&1 && return 0
    kill -0 "$1" 2>/dev/null || return 1     # proses mati -> berhenti menunggu
    sleep 5
  done
  return 1
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r TAG MID XFLAGS <<< "$entry"
  OUT="ann_${TAG}.jsonl"

  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] $TAG  <-  $MID"
  echo "  flag: $XFLAGS"

  # shellcheck disable=SC2086
  vllm serve "$MID" \
      --port "$PORT" \
      --dtype bfloat16 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.92 \
      $XFLAGS \
      > "vllm_${TAG}.log" 2>&1 &
  VPID=$!

  if ! wait_ready "$VPID"; then
    echo "  !! server $TAG gagal siap. 20 baris terakhir log:"
    tail -20 "vllm_${TAG}.log" | sed 's/^/     /'
    kill $VPID 2>/dev/null; wait $VPID 2>/dev/null; continue
  fi
  echo "[$(date +%H:%M:%S)] server siap"

  # probe dulu: 'output invalid' >5% -> turunkan CHUNK; throughput -> ETA sebenarnya
  python3 probe.py --base-url "$BASE" --model "$MID" --chunk "$CHUNK" \
          --concurrency 16 --calls 16 2>&1 | tee "probe_${TAG}.txt"

  python3 annotate.py --tag "$TAG" --base-url "$BASE" --model "$MID" \
          --chunk "$CHUNK" --concurrency "$CONC" --out "$OUT"

  kill $VPID 2>/dev/null; wait $VPID 2>/dev/null
  echo "[$(date +%H:%M:%S)] $TAG selesai -> $(wc -l < "$OUT" 2>/dev/null || echo 0) baris"
done

echo "=============================================================="
python3 agreement.py ann_qwen.jsonl ann_mistral.jsonl ann_yi.jsonl \
        | tee agreement_report.txt
