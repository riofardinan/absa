#!/usr/bin/env bash
# Anotasi 5 model via Ollama — jalan di Windows NATIVE (tanpa WSL2) atau Linux.
#
# Persiapan:
#   1) Pasang Ollama, lalu di terminal terpisah:
#        (Windows PowerShell)  $env:OLLAMA_NUM_PARALLEL=4 ; ollama serve
#        (Linux)               OLLAMA_NUM_PARALLEL=4 ollama serve
#   2) ./run_all_ollama.sh --pull      # unduh kelima model dulu (~12 GB)
#   3) ./run_all_ollama.sh --pilot     # 500 unit per model, ~30 menit, PILIH 3 TERBAIK
#   4) ./run_all_ollama.sh             # produksi
set -uo pipefail

HOST=${HOST:-http://localhost:11434}
CHUNK=${CHUNK:-25}
CONC=${CONC:-4}          # samakan dengan OLLAMA_NUM_PARALLEL
NUMCTX=${NUMCTX:-6144}   # prompt ~3.800 tok; default Ollama 2048-4096 MEMOTONG diam-diam
WAVE=${WAVE:-200}
CSV=${CSV:-../fintech_reviews_curated.csv}

# tag_kita|tag_ollama  — LIMA keluarga pretraining berbeda (ganjil: tanpa seri 2-2)
MODELS=(
  "qwen35|qwen3.5:4b"       # Alibaba, 2026
  "gemma4|gemma4:e2b"       # Google,  2026
  "granite4|granite4:3b"    # IBM,     2025/26
  "phi4|phi4-mini"          # Microsoft
  "llama32|llama3.2:3b"     # Meta  (alternatif: falcon3:3b, TII)
)

if [ "${1:-}" = "--pull" ]; then
  for e in "${MODELS[@]}"; do echo "== ${e#*|}"; ollama pull "${e#*|}"; done; exit 0
fi
LIMIT=""; [ "${1:-}" = "--pilot" ] && LIMIT="--limit 500 --wave 20"

for e in "${MODELS[@]}"; do
  IFS='|' read -r TAG MID <<< "$e"
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] $TAG  <-  $MID"
  # shellcheck disable=SC2086
  python3 annotate_ollama.py --tag "$TAG" --model "$MID" --host "$HOST" \
      --chunk "$CHUNK" --concurrency "$CONC" --num-ctx "$NUMCTX" --wave "$WAVE" $LIMIT \
    || { echo "  !! $TAG GAGAL — lanjut ke model berikutnya"; continue; }
  echo "[$(date +%H:%M:%S)] $TAG selesai -> $(wc -l < "ann_${TAG}.jsonl" 2>/dev/null || echo 0) baris"
done

echo "=============================================================="
if [ -n "$LIMIT" ]; then
  echo "### PILOT — pilih 3 dari 5 (fail% < 1, over-label terendah, KELUARGA BERBEDA)"
  python3 compare_models.py ann_qwen35.jsonl ann_gemma4.jsonl ann_granite4.jsonl \
                            ann_phi4.jsonl ann_llama32.jsonl
else
  for e in "${MODELS[@]}"; do
    TAG="${e%%|*}"; [ -s "ann_${TAG}.jsonl" ] && python3 export.py --tag "$TAG" --csv "$CSV"
  done
  echo "### Ganti daftar di bawah dengan 3 model yang kamu pilih dari pilot:"
  python3 agreement.py ann_qwen35.jsonl ann_gemma4.jsonl ann_phi4.jsonl | tee agreement_ollama.txt
fi
