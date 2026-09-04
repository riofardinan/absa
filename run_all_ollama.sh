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

# Windows/Git Bash memakai `python`, bukan `python3` (yang malah tertangkap alias
# Microsoft Store). Deteksi interpreter yang benar-benar bisa dijalankan.
PY=""
for c in python3 python py; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys" >/dev/null 2>&1; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || { echo "!! Python tidak ditemukan (coba python3/python/py)"; exit 1; }
echo "python: $PY ($($PY -V 2>&1))"

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
  "$PY" ollama_pull.py --host "$HOST" $(for e in "${MODELS[@]}"; do printf "%s " "${e#*|}"; done); exit 0
fi
LIMIT=""; [ "${1:-}" = "--pilot" ] && LIMIT="--limit 500 --wave 20"

for e in "${MODELS[@]}"; do
  IFS='|' read -r TAG MID <<< "$e"
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] $TAG  <-  $MID"
  # shellcheck disable=SC2086
  "$PY" annotate_ollama.py --tag "$TAG" --model "$MID" --host "$HOST" \
      --chunk "$CHUNK" --concurrency "$CONC" --num-ctx "$NUMCTX" --wave "$WAVE" $LIMIT \
    || { echo "  !! $TAG GAGAL — lanjut ke model berikutnya"; continue; }
  echo "[$(date +%H:%M:%S)] $TAG selesai -> $(wc -l < "ann_${TAG}.jsonl" 2>/dev/null || echo 0) baris"
done

echo "=============================================================="
if [ -n "$LIMIT" ]; then
  echo "### PILOT — pilih 3 dari 5 (fail% < 1, over-label terendah, KELUARGA BERBEDA)"
  "$PY" compare_models.py ann_qwen35.jsonl ann_gemma4.jsonl ann_granite4.jsonl \
                            ann_phi4.jsonl ann_llama32.jsonl
else
  for e in "${MODELS[@]}"; do
    TAG="${e%%|*}"; [ -s "ann_${TAG}.jsonl" ] && "$PY" export.py --tag "$TAG" --csv "$CSV"
  done
  ARGS=""; for e in "${MODELS[@]}"; do
    T="${e%%|*}"; [ -s "ann_${T}.jsonl" ] && ARGS="$ARGS ann_${T}.jsonl"
  done
  echo "### AGREEMENT 5 MODEL (ganjil -> tidak ada seri 2-2)"
  # shellcheck disable=SC2086
  "$PY" agreement.py $ARGS | tee agreement_5model.txt
  echo
  echo "### PERBANDINGAN KUALITAS"
  # shellcheck disable=SC2086
  "$PY" compare_models.py $ARGS | tee compare_5model.txt
  echo
  echo "Subset 3 model bisa dihitung kapan saja tanpa anotasi ulang, mis.:"
  echo "  $PY agreement.py ann_qwen35.jsonl ann_gemma4.jsonl ann_granite4.jsonl"
fi
