#!/usr/bin/env bash
# Jalankan DI SERVER A100. Menentukan plafon vLLM/CUDA yang sebenarnya.
echo "=== GPU & DRIVER ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
echo
echo "=== CUDA runtime tertinggi yang didukung driver ==="
nvidia-smi | grep -i "CUDA Version"
echo
echo "=== PYTHON / TORCH ==="
python3 -c "import sys; print('python', sys.version.split()[0])"
python3 - <<'PY' 2>&1 | tail -12
try:
    import torch
    print("torch      :", torch.__version__)
    print("built cuda :", torch.version.cuda)
    print("gpu avail  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"device     : {p.name}  {p.total_memory/2**30:.1f} GiB  sm_{p.major}{p.minor}")
        # tes kernel nyata, bukan sekadar is_available()
        import torch as t
        a = t.randn(2048, 2048, device="cuda", dtype=t.bfloat16)
        print("bf16 matmul:", "OK" if (a @ a).isfinite().all().item() else "GAGAL")
except ImportError:
    print("torch BELUM terpasang")
except Exception as e:
    print("torch ERROR:", type(e).__name__, e)
PY
echo
echo "=== VLLM ==="
python3 -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -2
echo
echo "=== DIAGNOSIS ==="
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
echo "driver major: $DRV"
if   [ "$DRV" -ge 570 ]; then echo "-> vLLM terbaru (0.27+) OK. Model 2026 bisa dipakai."
elif [ "$DRV" -ge 550 ]; then echo "-> vLLM ~0.9-0.10 (cu124). Gemma-3 / Qwen3 OK. Gemma-4 / Nemotron-3.5 kemungkinan TIDAK."
elif [ "$DRV" -ge 525 ]; then echo "-> vLLM cu121 (~0.6.x-0.9.x). AMAN: Llama-3.1, Qwen2.5, Gemma-2, Mistral, Phi-4."
                              echo "   TIDAK: Gemma-4, Qwen3.6/3.8, Nemotron-3.5 (arsitektur baru)."
else echo "-> driver terlalu lama; minta admin update."
fi
echo
echo "Pemasangan yang cocok untuk driver 525 (CUDA 12.1):"
echo "  pip install 'torch==2.4.0' --index-url https://download.pytorch.org/whl/cu121"
echo "  pip install 'vllm==0.6.3.post1'"
echo "  # kalau gagal, coba vllm==0.5.4 atau 0.4.2"
