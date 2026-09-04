#!/usr/bin/env python3
"""Anotasi via Ollama — alternatif vLLM. Jalan di Windows NATIVE (tanpa WSL2).

Memakai prompt.py / parsing.py / ontology.py yang sama, jadi output ann_*.jsonl
identik formatnya dan langsung bisa dipakai agreement.py + export.py.

  ollama pull qwen3.5:4b
  python3 annotate_ollama.py --tag qwen35 --model qwen3.5:4b --limit 500

PENTING: Ollama default num_ctx-nya kecil (2048/4096) dan akan MEMOTONG prompt
~3.800 token secara diam-diam. Skrip ini mengirim num_ctx eksplisit.
"""
import argparse, collections, json, os, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ontology import PROMPT_VERSION
from prompt import build_batch
from parsing import parse_chunk, parse_chunk_compact, parse_chunk_partial

_lk = threading.Lock()


def call(prompt, a):
    """Ollama native /api/chat — OpenAI-compat endpoint tidak menerima num_ctx."""
    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": a.thinking,                 # matikan reasoning bila model mendukung
        "keep_alive": a.keep_alive,
        # Tiap model Ollama membawa PARAMETER sendiri dari Modelfile-nya
        # (top_k, top_p, repeat_penalty, ...). Menyetel temperature saja TIDAK
        # membuat perbandingan apple-to-apple: repeat_penalty khususnya tetap
        # mengubah output walau temperature 0. Semua dipatok eksplisit.
        "options": {
            "temperature": 0.0,        # greedy, satu pass (bukan self-consistency)
            "top_p": 1.0,              # nonaktif
            "top_k": 0,                # nonaktif
            "repeat_penalty": 1.0,     # netral: 1.1 default sebagian model merusak
            "repeat_last_n": 0,        # format kita memang repetitif; jangan dihukum
            "seed": a.seed,            # determinisme & reproduktibilitas
            "num_ctx": a.num_ctx,      # WAJIB: default Ollama memotong prompt diam-diam
            "num_predict": a.max_tokens,
            "stop": [],                # buang stop token bawaan tiap Modelfile
        },
    }
    req = urllib.request.Request(
        a.host.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    delay = 2.0
    for k in range(a.max_retries):
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                j = json.loads(r.read())
            return (j.get("message") or {}).get("content", "")
        except urllib.error.HTTPError as e:
            msg = e.read()[:300].decode("utf-8", "replace")
            if "does not support thinking" in msg and body.get("think") is not None:
                body.pop("think")            # model tanpa mode thinking
                req.data = json.dumps(body).encode()
                continue
            if k == a.max_retries - 1:
                raise RuntimeError(f"HTTP {e.code}: {msg}")
        except Exception:
            if k == a.max_retries - 1:
                raise
        time.sleep(delay); delay = min(delay * 2, 30)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True, help="nama tag Ollama, mis. qwen3.5:4b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--work", default="work.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--wave", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="samakan dengan OLLAMA_NUM_PARALLEL di server")
    ap.add_argument("--num-ctx", type=int, default=6144)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = chunk*25+200")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-alive", default="30m")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--thinking", action="store_true",
                    help="biarkan reasoning aktif (default: dimatikan)")
    ap.add_argument("--format", choices=["compact", "json"], default="compact")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--abort-above", type=float, default=0.5)
    a = ap.parse_args()
    a.out = a.out or f"ann_{a.tag}.jsonl"
    a.thinking = True if a.thinking else False
    if not a.max_tokens:
        a.max_tokens = (a.chunk * 25 + 200) if a.format == "compact" else (a.chunk * 130 + 300)
    compact = a.format == "compact"

    units = [json.loads(l) for l in open(a.work, encoding="utf-8")]
    if a.limit:
        units = units[:a.limit]
    chunks = [units[i:i + a.chunk] for i in range(0, len(units), a.chunk)]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try: done.add(json.loads(l)["chunk"])
            except Exception: pass
    todo = [(i, c) for i, c in enumerate(chunks) if i not in done]
    print(f"[{a.tag}] {len(units):,} unit -> {len(chunks):,} chunk @ {a.chunk} | "
          f"selesai {len(done):,} | sisa {len(todo):,}")
    print(f"  ollama: {a.model} @ {a.host} | num_ctx={a.num_ctx} "
          f"num_predict={a.max_tokens} think={a.thinking} concurrency={a.concurrency}")
    print(f"  sampling DIPATOK SAMA untuk semua model: temperature=0.0 top_p=1.0 "
          f"top_k=0 repeat_penalty=1.0 repeat_last_n=0 seed={a.seed} stop=[]",
          flush=True)
    if not todo:
        print("  sudah lengkap."); return

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time(); st = {"unit": 0, "retry": 0, "fail": 0, "dbg": 0}
    fmode = collections.Counter()
    fh = open(a.out, "a", encoding="utf-8")

    def gen(prompts):
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            return list(ex.map(lambda p: call(p, a), prompts))

    for w0 in range(0, len(todo), a.wave):
        wave = todo[w0:w0 + a.wave]
        res, holes = {}, []
        outs = gen([build_batch([u["text"] for u in c]) for _, c in wave])
        for (idx, cu), txt in zip(wave, outs):
            if a.debug and st["dbg"] < a.debug:
                st["dbg"] += 1
                print(f"\n----- OUTPUT MENTAH #{st['dbg']} ({len(txt)} char) -----\n"
                      f"{txt}\n-----", flush=True)
            r, miss = parse_chunk_partial(txt, len(cu), compact)
            if miss:
                fmode[f"{len(cu)-len(miss)}/{len(cu)} baris terbaca"] += 1
            res[idx] = r
            holes += [(idx, m - 1, cu[m - 1]) for m in miss]

        if holes:                                  # ulangi HANYA unit yang bolong
            st["retry"] += len(holes)
            outs = gen([build_batch([u["text"]]) for _, _, u in holes])
            for (idx, pos, _u), txt in zip(holes, outs):
                one, _ = parse_chunk_partial(txt, 1, compact)
                res[idx][pos] = one[0]

        lines = []
        for idx, cu in wave:
            for u, v in zip(cu, res[idx]):
                bad = v is None
                st["fail"] += bad
                labels, exc, flags = ([], None, []) if bad else v
                lines.append(json.dumps({
                    "uid": u["uid"], "chunk": idx, "labels": labels,
                    "exclusion": exc, "flags": flags,
                    "method": "LLM", "provider": "ollama", "model_name": a.model,
                    "model_tag": a.tag, "prompt_version": PROMPT_VERSION,
                    "wire_format": a.format, "chunk_size": a.chunk,
                    "temperature": 0.0, "top_p": 1.0, "top_k": 0,
                    "repeat_penalty": 1.0, "seed": a.seed,
                    "num_ctx": a.num_ctx, "num_predict": a.max_tokens,
                    "think": a.thinking,
                    "parser_valid": not bad, "annotated_at": ts,
                }, ensure_ascii=False))
                st["unit"] += 1
        fh.write("\n".join(lines) + "\n"); fh.flush(); os.fsync(fh.fileno())

        if w0 == 0 and st["unit"] and st["fail"] / st["unit"] > a.abort_above:
            print(f"\n!! BERHENTI: {100*st['fail']/st['unit']:.1f}% gagal parse di gelombang "
                  f"pertama. Jalankan ulang dgn --debug 2.\n"
                  f"   Tersering: num_ctx kurang (prompt terpotong), reasoning aktif, "
                  f"atau num_predict kurang.", flush=True)
            fh.close(); raise SystemExit(2)

        el = time.time() - t0
        d = min(w0 + a.wave, len(todo))
        print(f"  {d:,}/{len(todo):,} chunk | {st['unit']:,} unit | {el/60:5.1f} mnt "
              f"| ETA {(len(todo)-d)/(d/el)/60:6.1f} mnt | {st['unit']/el:5.1f} unit/dtk "
              f"| retry {st['retry']} fail {st['fail']}", flush=True)

    fh.close()
    el = (time.time() - t0) / 60
    if fmode:
        print("\n  MODE KEGAGALAN:")
        for m, c in fmode.most_common(6):
            print(f"    {c:6,}  {m}")
    print(f"[{a.tag}] SELESAI {st['unit']:,} unit / {el:.1f} menit "
          f"({st['unit']/el/60:.1f} unit/dtk) | retry {st['retry']} "
          f"fail {st['fail']} ({100*st['fail']/max(st['unit'],1):.2f}%)")


if __name__ == "__main__":
    main()
