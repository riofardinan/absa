#!/usr/bin/env python3
"""Uji cudaMallocAsync di MIG dengan budget REALISTIS (bukan 34 GiB seperti tes lama).

  python3 test_alloc.py
"""
import os, subprocess, sys

CHILD = r'''
import torch, sys
GiB = 1024**3
budget = float(sys.argv[1])
w = 14.25                                   # bobot Qwen2.5-7B (dari log-mu)
kv = budget - w - 1.6                       # sisa utk KV cache (aktivasi ~1.44 + non-torch 0.13)
try:
    hold = torch.zeros(int(w*GiB//2), dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    cache, n = [], 28
    for _ in range(n):
        cache.append(torch.zeros(int((kv/n)*GiB//2), dtype=torch.float16, device="cuda"))
    torch.cuda.synchronize()
    free, tot = torch.cuda.mem_get_info()
    print(f"OK   dipakai {(w+kv):5.1f} GiB   sisa bebas {free/GiB:5.1f} GiB")
except Exception as e:
    print(f"GAGAL  {type(e).__name__}: {str(e)[:120]}")
'''

env = dict(os.environ)
env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
print("backend:cudaMallocAsync — mencari budget terbesar yang muat\n")
print("  budget  hasil")
for b in (34, 32, 31, 30, 28, 26, 24):
    r = subprocess.run([sys.executable, "-c", CHILD, str(b)], env=env,
                       capture_output=True, text=True, timeout=300)
    out = (r.stdout.strip().splitlines() or ["CRASH: " +
           (r.stderr.strip().splitlines() or [""])[-1][:120]])[-1]
    print(f"  {b:>4} GiB  {out}")
    if out.startswith("OK"):
        util = b / 39.39
        print(f"\n  ==> PAKAI --gpu-util {util:.2f}  (= {b} GiB dari 39,39 GiB)")
        print(f"      PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync")
        break
else:
    print("\n  Semua gagal -> turunkan versi:  pip install vllm==0.6.3.post1")
    print("  (menarik torch 2.4.0+cu121; allocator-nya belum punya panggilan NVML itu)")
