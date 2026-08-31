#!/usr/bin/env bash
# Jalankan anotasi di dalam image resmi vLLM — menghindari neraka dependensi
# torch/CUDA/FlashInfer di WSL2.
#
# Prasyarat (sekali saja):
#   - Docker Desktop dgn backend WSL2, "Use the WSL 2 based engine" + GPU support ON
#   - uji: docker run --rm --gpus all vllm/vllm-openai:v0.28.0 nvidia-smi
#
# PENTING: PROJECT dan HF_CACHE harus di filesystem LINUX (~/...), JANGAN /mnt/c.
# Bind mount dari /mnt/c bahkan lebih lambat daripada akses langsung.
set -uo pipefail

IMAGE=${IMAGE:-vllm/vllm-openai:v0.28.0}
PROJECT=${PROJECT:-$HOME/absa}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}

mkdir -p "$HF_CACHE"
[ -f "$PROJECT/annotate_offline.py" ] || {
  echo "!! $PROJECT/annotate_offline.py tidak ada."
  echo "   Salin proyek ke filesystem Linux dulu:"
  echo "     mkdir -p ~/absa && cp -r /mnt/c/Users/User/absa/* ~/absa/"
  exit 1; }

exec docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  -v "$PROJECT":/work \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -w /work \
  -e VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  --entrypoint python3 \
  "$IMAGE" \
  annotate_offline.py "$@"
