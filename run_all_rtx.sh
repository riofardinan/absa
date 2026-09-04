#!/usr/bin/env bash
# Anotasi SEQUENTIAL di RTX 5070 12GB (Blackwell sm_120), jalur in-process.
# Cohort KEDUA — tag sengaja berbeda dari run A100 supaya keduanya bisa
# dibandingkan langsung oleh agreement.py.
#
# Setup (di dalam WSL2, bukan Windows native):
#   "$PY" -m venv .venv && source .venv/bin/activate
#   pip install -U pip
#   pip install vllm==0.28.0        # menarik torch 2.13.x sendiri; JANGAN pasang torch duluan
#   "$PY" -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
#   # harus mencetak (12, 0) untuk RTX 5070
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

CHUNK=${CHUNK:-25}
MML=${MML:-4608}
WAVE=${WAVE:-1000}
CSV=${CSV:-../fintech_reviews_curated.csv}
# gpu-util, kv-cache-dtype, dan CUDA graph ditentukan OTOMATIS dari GPU terdeteksi.
# Override hanya bila perlu: GU=0.80 EXTRA="--kv-cache-dtype auto"
GU=${GU:-0}
EXTRA=${EXTRA:-}

# tag|model_id — GANJIL (3 atau 5). Jumlah genap menciptakan seri 2-2 yang
# tidak punya mayoritas dan meledakkan antrian adjudikasi manusia.
MODELS=(
  "qwen35|cyankiwi/Qwen3.5-4B-AWQ-4bit"
  "nemotron|nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"
  "gemma4|cyankiwi/gemma-4-E2B-it-AWQ-INT4"
)
# Opsi 5 model (tambahkan dua baris di atas):
#   "ministral|cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit"
#   "phi|microsoft/Phi-4-mini-instruct"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r TAG MID <<< "$entry"
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] $TAG  <-  $MID"

  # shellcheck disable=SC2086
  "$PY" annotate_offline.py --tag "$TAG" --model "$MID" \
      --chunk "$CHUNK" --max-model-len "$MML" --wave "$WAVE" \
      ${GU:+--gpu-util "$GU"} $EXTRA \
    || { echo "  !! $TAG GAGAL — lanjut ke model berikutnya"; continue; }

  echo "[$(date +%H:%M:%S)] $TAG selesai -> $(wc -l < "ann_${TAG}.jsonl" 2>/dev/null || echo 0) baris"
done

echo "=============================================================="
for entry in "${MODELS[@]}"; do
  TAG="${entry%%|*}"
  [ -s "ann_${TAG}.jsonl" ] && "$PY" export.py --tag "$TAG" --csv "$CSV"
done

echo "=============================================================="
echo "### Cohort RTX (3 model generasi 2025-2026, 3-4B)"
"$PY" agreement.py ann_qwen35.jsonl ann_nemotron.jsonl ann_gemma4.jsonl \
  | tee agreement_rtx.txt
