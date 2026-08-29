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
from datetime import datetime, timezone

# HARUS diset SEBELUM torch diimpor (vllm mengimpor torch). Export di shell sering
# tidak berlaku kalau torch sudah terimpor lebih dulu di proses lain.
# expandable_segments memakai VMM API + NVML -> pemicu NVML_SUCCESS assert di
# driver/container yang NVML-nya terbatas.
# MIG (7g.40gb) memblokir sebagian API NVML dan CUDA VMM. Caching allocator
# PyTorch 2.5 memanggil NVML -> "NVML_SUCCESS == r INTERNAL ASSERT FAILED".
# cudaMallocAsync memakai allocator native CUDA, jalur NVML itu tidak disentuh.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

from ontology import PROMPT_VERSION
from prompt import build_batch
from parsing import parse_chunk, parse_chunk_compact, extract_json


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
    _tpl = getattr(tok, "chat_template", None) is not None
    def _wrap(p):
        return tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True) if _tpl else p
    print(f"chat tpl   : {'ADA (overhead sudah dihitung)' if _tpl else 'TIDAK ADA'}")
    units, chunks = load_chunks(a)
    import random
    random.seed(0)
    sample = random.sample(chunks[:-1] or chunks, min(200, len(chunks)))
    lens = [len(tok(_wrap(build_batch([u["text"] for u in c]))).input_ids) for c in sample]
    lens.sort()
    est_out = a.max_tokens
    p50, p99, mx = lens[len(lens)//2], lens[int(len(lens)*.99)], lens[-1]
    print(f"model      : {a.model}")
    print(f"chunk      : {a.chunk}   (sampel {len(lens)} chunk)")
    print(f"input tok  : median {p50:,}  p99 {p99:,}  max {mx:,}")
    print(f"output est : {est_out:,} tok  (max-tokens; terukur ~100 tok/review)")
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
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0 = otomatis: chunk x 130 + 300 (terukur ~100 tok/review)")
    ap.add_argument("--gpu-util", type=float, default=0.60)   # MIG: usable ~24 GiB dari 39,39
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-prefix-caching", action="store_true",
                    help="pakai kalau model punya sliding window (vLLM 0.6.x menolak kombinasi itu)")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="AKTIFKAN CUDA graph. Default OFF: cudaMallocAsync (wajib di MIG) "
                         "tidak kompatibel dengan graph capture -> 'uncaptured free of a "
                         "captured allocation' lalu CUDA error: invalid argument.")
    ap.add_argument("--provider", default="vllm-local",
                    help="dicatat sbg provenance (§8): vllm-local, openai, dst.")
    ap.add_argument("--format", choices=["compact", "json"], default="compact",
                    help="compact = 1 baris/review, ~10x lebih sedikit token output. "
                         "json dipertahankan untuk uji validasi format.")
    ap.add_argument("--debug-fail", type=int, default=0,
                    help="cetak N output yang GAGAL parse — diagnosis fail%% tinggi")
    ap.add_argument("--debug", type=int, default=0,
                    help="cetak N output mentah pertama — pakai kalau fail%% tinggi")
    ap.add_argument("--check-tokens", action="store_true",
                    help="hanya ukur panjang token via tokenizer, tidak memuat model ke GPU")
    a = ap.parse_args()
    a.out = a.out or f"ann_{a.tag}.jsonl"

    parse = parse_chunk_compact if a.format == "compact" else parse_chunk
    if not a.max_tokens:
        a.max_tokens = (a.chunk * 25 + 200) if a.format == "compact" else (a.chunk * 130 + 300)

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
              enforce_eager=not a.cuda_graph,
              seed=a.seed, disable_log_stats=True)
    sp = SamplingParams(temperature=0, max_tokens=a.max_tokens)   # 1 pass -> greedy

    # WAJIB: model *-Instruct butuh chat template (ChatML dsb). llm.generate() TIDAK
    # menerapkannya otomatis -> tanpa ini model melanjutkan teks, bukan menjawab
    # instruksi, dan hampir semua output gagal di-parse.
    tok = llm.get_tokenizer()
    has_tpl = getattr(tok, "chat_template", None) is not None
    print(f"  chat template: {'ADA -> dipakai' if has_tpl else 'TIDAK ADA -> prompt mentah'}",
          flush=True)

    def wrap(p):
        if not has_tpl:
            return p
        return tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)

    dbg = {"n": 0, "f": 0}

    def gen(prompts, sizes=None):
        outs = [o.outputs[0].text for o in llm.generate([wrap(p) for p in prompts],
                                                        sp, use_tqdm=False)]
        if a.debug and dbg["n"] < a.debug:
            for t in outs[:a.debug - dbg["n"]]:
                dbg["n"] += 1
                print(f"\n----- OUTPUT MENTAH #{dbg['n']} ({len(t)} char) -----\n{t}\n-----",
                      flush=True)
        if a.debug_fail and sizes:                     # cetak hanya yang GAGAL parse
            for t, n in zip(outs, sizes):
                if dbg["f"] >= a.debug_fail: break
                if parse(t, n) is None:
                    dbg["f"] += 1
                    got = len(t.strip().splitlines())
                    print(f"\n----- GAGAL #{dbg['f']}: minta {n} objek, dapat {got}, "
                          f"{len(t)} char -----\n{t[:4000]}\n-----", flush=True)
        return outs

    prov_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time(); nrep = nsplit = nfail = nunit = 0
    fh = open(a.out, "a", encoding="utf-8")

    for w0 in range(0, len(todo), a.wave):
        wave = todo[w0:w0 + a.wave]
        res = {}

        # --- lintasan 1 ---------------------------------------------------
        outs = gen([build_batch([u["text"] for u in c]) for _, c in wave],
                   [len(c) for _, c in wave])
        redo = []
        for (idx, cu), txt in zip(wave, outs):
            p = parse(txt, len(cu))
            if p is None: redo.append((idx, cu))
            else: res[idx] = p

        # --- lintasan 2: repair call (Wittlinger: "second LLM repair call") -
        if redo:
            nrep += len(redo)
            outs = gen([build_batch([u["text"] for u in c]) for _, c in redo],
                       [len(c) for _, c in redo])
            still = []
            for (idx, cu), txt in zip(redo, outs):
                p = parse(txt, len(cu))
                if p is None: still.append((idx, cu))
                else: res[idx] = p

            # --- lintasan 3: pecah ke single ------------------------------
            if still:
                nsplit += len(still)
                flat = [(idx, u) for idx, cu in still for u in cu]
                outs = gen([build_batch([u["text"]]) for _, u in flat], [1]*len(flat))
                per = {}
                for (idx, _u), txt in zip(flat, outs):
                    p = parse(txt, 1)
                    per.setdefault(idx, []).append(p[0] if p else ([], "ERROR_PARSE", []))
                for idx, cu in still:
                    res[idx] = per.get(idx, [([], "ERROR_PARSE", [])] * len(cu))

        # --- tulis + checkpoint -------------------------------------------
        lines = []
        for idx, cu in wave:
            r = res.get(idx) or [([], "ERROR_PARSE", [])] * len(cu)   # jaring pengaman
            for u, (labels, exc, flags) in zip(cu, r):
                bad = exc == "ERROR_PARSE"
                nfail += bad
                lines.append(json.dumps({
                    "uid": u["uid"], "chunk": idx,
                    "labels": labels, "exclusion": None if bad else exc,
                    "flags": flags,
                    # provenance (§8, §11)
                    "method": "LLM", "provider": a.provider, "model_name": a.model,
                    "model_tag": a.tag, "prompt_version": PROMPT_VERSION,
                    "wire_format": a.format, "chunk_size": a.chunk,
                    "temperature": 0, "parser_valid": not bad,
                    "annotated_at": prov_ts,
                }, ensure_ascii=False))
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
