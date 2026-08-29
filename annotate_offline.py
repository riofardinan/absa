#!/usr/bin/env python3
"""Anotasi OFFLINE via vLLM LLM.generate() — in-process.

Tanpa HTTP server, tanpa subprocess engine, tanpa ZMQ: seluruh jalur yang gagal
dengan NVML_SUCCESS assert pada `vllm serve` tidak dilewati. Untuk job batch juga
lebih cepat — tidak ada overhead HTTP, dan scheduler vLLM melihat seluruh
gelombang prompt sekaligus sehingga menyusun batch-nya sendiri secara optimal.

  # 1) cek panjang token dengan tokenizer ASLI dulu (tanpa muat model ke GPU)
  python3 annotate_offline.py --tag qwen --model Qwen/Qwen2.5-7B-Instruct --check-tokens

  # 2) pilot
  python3 annotate_offline.py --tag qwen --model Qwen/Qwen2.5-7B-Instruct --limit 2000

  # 3) produksi
  python3 annotate_offline.py --tag qwen --model Qwen/Qwen2.5-7B-Instruct

Aman diputus: jalankan ulang, chunk yang sudah tertulis dilewati.
"""
import argparse, json, os, time

# HARUS diset SEBELUM torch diimpor (vllm mengimpor torch). Export di shell sering
# tidak berlaku kalau torch sudah terimpor lebih dulu di proses lain.
# expandable_segments memakai VMM API + NVML -> pemicu NVML_SUCCESS assert di
# driver/container yang NVML-nya terbatas.
# MIG (7g.40gb) memblokir sebagian API NVML dan CUDA VMM. Caching allocator
# PyTorch 2.5 memanggil NVML -> "NVML_SUCCESS == r INTERNAL ASSERT FAILED".
# cudaMallocAsync memakai allocator native CUDA, jalur NVML itu tidak disentuh.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

from prompt import build_batch
from parsing import parse_chunk       # parser + validator bersama


# ------------------------------------------------------------------ util
def load_chunks(a):
    units = [json.loads(l) for l in open(a.work, encoding="utf-8")]
    if a.limit:
        units = units[:a.limit]
    return units, [units[i:i + a.chunk] for i in range(0, len(units), a.chunk)]


def check_tokens(a):
    """Ukur panjang prompt dengan tokenizer asli. Menjawab: muat di max_model_len?"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    units, chunks = load_chunks(a)
    import random
    random.seed(0)
    sample = random.sample(chunks[:-1] or chunks, min(200, len(chunks)))
    lens = [len(tok(build_batch([u["text"] for u in c])).input_ids) for c in sample]
    lens.sort()
    est_out = a.chunk * 40
    p50, p99, mx = lens[len(lens)//2], lens[int(len(lens)*.99)], lens[-1]
    print(f"model      : {a.model}")
    print(f"chunk      : {a.chunk}   (sampel {len(lens)} chunk)")
    print(f"input tok  : median {p50:,}  p99 {p99:,}  max {mx:,}")
    print(f"output est : {est_out:,} tok  (~40 tok JSON per review)")
    print(f"TOTAL max  : {mx + est_out:,} tok   vs --max-model-len {a.max_model_len:,}")
    head = mx + est_out
    if head > a.max_model_len:
        print(f"\n  !! MELAMPAUI. Naikkan --max-model-len ke >= {((head//512)+1)*512:,}, "
              f"atau turunkan --chunk.")
    else:
        print(f"\n  OK, sisa headroom {a.max_model_len - head:,} tok.")
    return head <= a.max_model_len


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", default="work.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--wave", type=int, default=1000,
                    help="chunk per gelombang sebelum checkpoint (kecil = rugi lebih sedikit saat crash)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.60)   # MIG: usable ~24 GiB dari 39,39
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-prefix-caching", action="store_true",
                    help="pakai kalau model punya sliding window (vLLM 0.6.x menolak kombinasi itu)")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="matikan CUDA graph — coba ini kalau driver lama bermasalah")
    ap.add_argument("--check-tokens", action="store_true",
                    help="hanya ukur panjang token via tokenizer, tidak memuat model ke GPU")
    a = ap.parse_args()
    a.out = a.out or f"ann_{a.tag}.jsonl"

    if a.check_tokens:
        raise SystemExit(0 if check_tokens(a) else 1)

    units, chunks = load_chunks(a)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try: done.add(json.loads(l)["chunk"])
            except Exception: pass
    todo = [(i, c) for i, c in enumerate(chunks) if i not in done]
    print(f"[{a.tag}] {len(units):,} unit -> {len(chunks):,} chunk @ {a.chunk}"
          f" | selesai {len(done):,} | sisa {len(todo):,}", flush=True)
    if not todo:
        print("  sudah lengkap."); return

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16",
              max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_util,
              enable_prefix_caching=not a.no_prefix_caching,
              enforce_eager=a.enforce_eager,
              seed=a.seed, disable_log_stats=True)
    sp = SamplingParams(temperature=0, max_tokens=a.max_tokens)   # 1 pass -> greedy

    def gen(prompts):
        return [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False)]

    t0 = time.time(); nrep = nsplit = nfail = nunit = 0
    fh = open(a.out, "a", encoding="utf-8")

    for w0 in range(0, len(todo), a.wave):
        wave = todo[w0:w0 + a.wave]
        res = {}

        # --- lintasan 1 ---------------------------------------------------
        outs = gen([build_batch([u["text"] for u in c]) for _, c in wave])
        redo = []
        for (idx, cu), txt in zip(wave, outs):
            p = parse_chunk(txt, len(cu))
            if p is None: redo.append((idx, cu))
            else: res[idx] = p

        # --- lintasan 2: repair call (Wittlinger: "second LLM repair call") -
        if redo:
            nrep += len(redo)
            outs = gen([build_batch([u["text"] for u in c]) for _, c in redo])
            still = []
            for (idx, cu), txt in zip(redo, outs):
                p = parse_chunk(txt, len(cu))
                if p is None: still.append((idx, cu))
                else: res[idx] = p

            # --- lintasan 3: pecah ke single ------------------------------
            if still:
                nsplit += len(still)
                flat = [(idx, u) for idx, cu in still for u in cu]
                outs = gen([build_batch([u["text"]]) for _, u in flat])
                per = {}
                for (idx, _u), txt in zip(flat, outs):
                    p = parse_chunk(txt, 1)
                    per.setdefault(idx, []).append(p[0] if p else ([], "ERROR_PARSE"))
                for idx, cu in still:
                    res[idx] = per.get(idx, [([], "ERROR_PARSE")] * len(cu))

        # --- tulis + checkpoint -------------------------------------------
        lines = []
        for idx, cu in wave:
            r = res.get(idx) or [([], "ERROR_PARSE")] * len(cu)   # jaring pengaman
            for u, (labels, exc) in zip(cu, r):
                if exc == "ERROR_PARSE": nfail += 1
                lines.append(json.dumps({"uid": u["uid"], "chunk": idx, "model": a.tag,
                                         "labels": labels, "exclusion": exc},
                                        ensure_ascii=False))
                nunit += 1
        fh.write("\n".join(lines) + "\n"); fh.flush(); os.fsync(fh.fileno())

        el = time.time() - t0
        d = min(w0 + a.wave, len(todo))
        print(f"  {d:,}/{len(todo):,} chunk | {nunit:,} unit | {el/60:5.1f} mnt "
              f"| ETA {(len(todo)-d)/(d/el)/60:6.1f} mnt "
              f"| {nunit/el:5.1f} unit/dtk "
              f"| repair {nrep} split {nsplit} fail {nfail}", flush=True)

    fh.close()
    el = (time.time() - t0) / 60
    print(f"[{a.tag}] SELESAI {nunit:,} unit / {el:.1f} menit "
          f"({nunit/el/60:.1f} unit/dtk) | repair {nrep} split {nsplit} "
          f"fail {nfail} ({100*nfail/max(nunit,1):.2f}%)")


if __name__ == "__main__":
    main()
