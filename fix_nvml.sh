#!/usr/bin/env bash
# Pinpoint NVML_SUCCESS assert. Jalankan DI SERVER, di conda env yang sama.
echo "=== 1. ENV yang memicu jalur NVML di allocator ==="
echo "PYTORCH_CUDA_ALLOC_CONF = '${PYTORCH_CUDA_ALLOC_CONF:-<kosong>}'"
echo "CUDA_VISIBLE_DEVICES    = '${CUDA_VISIBLE_DEVICES:-<kosong>}'"
echo "LD_LIBRARY_PATH         = '${LD_LIBRARY_PATH:-<kosong>}'"
echo
echo "=== 2. libnvidia-ml.so ganda? (stub conda menutupi driver) ==="
find / -name "libnvidia-ml.so*" -not -path "*/proc/*" 2>/dev/null | while read -r f; do
  printf "  %-70s -> %s\n" "$f" "$(readlink -f "$f" | xargs -r basename)"
done
echo "  ldconfig:"; ldconfig -p | grep -i nvidia-ml | sed 's/^/    /'
echo
echo "=== 3. NVML dari python ==="
python3 - <<'PY'
try:
    import pynvml as n; n.nvmlInit()
    print("  nvmlInit OK, driver:", n.nvmlSystemGetDriverVersion())
    h = n.nvmlDeviceGetHandleByIndex(0)
    print("  device:", n.nvmlDeviceGetName(h))
    try:
        n.nvmlDeviceGetComputeRunningProcesses_v3(h)
        print("  _v3 API: ADA")
    except Exception as e:
        print("  _v3 API: TIDAK ADA ->", type(e).__name__, e)
except Exception as e:
    print("  NVML GAGAL:", type(e).__name__, e)
PY
echo
echo "=== 4. Alokasi 16 GiB IN-PROCESS (seperti LLM.generate) ==="
python3 - <<'PY'
import torch, traceback
try:
    x = torch.zeros(16*1024**3//2, dtype=torch.float16, device="cuda")
    print("  IN-PROCESS: OK", x.numel()*2/2**30, "GiB")
    del x; torch.cuda.empty_cache()
except Exception as e:
    print("  IN-PROCESS: GAGAL ->", type(e).__name__, str(e)[:200])
PY
echo
echo "=== 5. Alokasi 16 GiB DI SPAWNED PROCESS (seperti vllm serve) ==="
python3 - <<'PY'
import multiprocessing as mp
def work(q):
    try:
        import torch
        x = torch.zeros(16*1024**3//2, dtype=torch.float16, device="cuda")
        q.put(("OK", x.numel()*2/2**30))
    except Exception as e:
        q.put(("GAGAL", f"{type(e).__name__}: {str(e)[:200]}"))
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    q = mp.Queue(); p = mp.Process(target=work, args=(q,)); p.start(); p.join(120)
    print("  SPAWNED:", *(q.get() if not q.empty() else ("TIMEOUT/CRASH","")))
PY
echo
echo "=== DIAGNOSIS ==="
echo "  4 OK + 5 GAGAL -> masalahnya spawn. Pakai annotate_offline.py (in-process),"
echo "                    atau tambahkan --disable-frontend-multiprocessing."
echo "  4 GAGAL & 5 GAGAL -> NVML/driver. Coba:"
echo "                    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False"
echo "                    dan pastikan libnvidia-ml.so dari DRIVER, bukan conda."
echo "  keduanya OK -> alokasi bukan penyebab; turunkan --gpu-memory-utilization ke 0.85."
