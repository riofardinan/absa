#!/usr/bin/env python3
"""Runner anotasi multi-item, stdlib saja, endpoint OpenAI-compatible.

Contoh:
  export API_KEY=...
  python3 annotate.py --tag modelA \
      --base-url https://api.provider.com/v1 --model nama-model \
      --chunk 25 --concurrency 24

Tahan mati di tengah jalan: jalankan ulang perintah yang sama, chunk yang sudah
selesai dilewati (checkpoint dari file output itu sendiri).
"""
import argparse, json, os, random, re, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

from ontology import ASPECTS, POLARITIES, EXCLUSIONS
from prompt import build_batch

VALID_A, VALID_P, VALID_E = set(ASPECTS), set(POLARITIES), set(EXCLUSIONS)
_lock = threading.Lock()
_stat = {"done": 0, "units": 0, "repair": 0, "split": 0, "fail": 0,
         "tin": 0, "tout": 0, "t0": time.time()}


# ------------------------------------------------------------------ HTTP
def call_api(prompt, args, key):
    body = {"model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,                 # 1 pass -> greedy, deterministik
            "max_tokens": args.max_tokens}
    if args.json_mode:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions", data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    delay = 2.0
    for attempt in range(args.max_retries):
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                j = json.loads(r.read())
            u = j.get("usage") or {}
            with _lock:
                _stat["tin"] += u.get("prompt_tokens", 0)
                _stat["tout"] += u.get("completion_tokens", 0)
            return j["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < args.max_retries - 1:
                ra = e.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.replace(".", "").isdigit()
                           else delay + random.uniform(0, 1))
                delay = min(delay * 2, 60)
                continue
            raise
        except Exception:
            if attempt < args.max_retries - 1:
                time.sleep(delay + random.uniform(0, 1)); delay = min(delay * 2, 60)
                continue
            raise
    raise RuntimeError("retry habis")


# ------------------------------------------------------------------ parsing
def extract_json(txt):
    t = re.sub(r"^\s*```(?:json)?|```\s*$", "", txt.strip(), flags=re.M).strip()
    for op, cl in (("[", "]"), ("{", "}")):
        i, k = t.find(op), t.rfind(cl)
        if i != -1 and k > i:
            try:
                v = json.loads(t[i:k + 1])
                return v if isinstance(v, list) else [v]
            except json.JSONDecodeError:
                pass
    return None


def norm_item(obj):
    """Kembalikan (labels, exclusion) tervalidasi, atau None kalau tak terselamatkan."""
    if not isinstance(obj, dict):
        return None
    labels, seen = [], set()
    for lb in (obj.get("labels") or []):
        if not isinstance(lb, dict):
            continue
        a = str(lb.get("aspect", "")).strip().upper()
        p = str(lb.get("polarity", "")).strip().upper()
        if a in VALID_A and p in VALID_P and a not in seen:
            seen.add(a); labels.append({"aspect": a, "polarity": p})
    exc = obj.get("exclusion")
    exc = str(exc).strip().upper() if exc not in (None, "", "null", "NULL") else None
    if exc is not None and exc not in VALID_E:
        exc = "OUT_OF_ONTOLOGY"
    if labels:
        exc = None                      # aturan: labels terisi -> exclusion null
    elif exc is None:
        return None                     # kosong tanpa alasan -> anggap gagal
    return labels, exc


def parse_chunk(txt, size):
    arr = extract_json(txt)
    if not isinstance(arr, list) or len(arr) != size:
        return None
    out = []
    for o in arr:
        r = norm_item(o)
        if r is None:
            return None
        out.append(r)
    return out


# ------------------------------------------------------------------ worker
def do_chunk(idx, units, args, key, fh):
    texts = [u["text"] for u in units]
    res = None
    try:
        res = parse_chunk(call_api(build_batch(texts), args, key), len(texts))
    except Exception:
        pass
    if res is None:                                    # repair call ke-2
        with _lock: _stat["repair"] += 1
        try:
            res = parse_chunk(call_api(build_batch(texts), args, key), len(texts))
        except Exception:
            pass
    if res is None:                                    # fallback: pecah ke single
        with _lock: _stat["split"] += 1
        res = []
        for t in texts:
            one = None
            try:
                one = parse_chunk(call_api(build_batch([t]), args, key), 1)
            except Exception:
                pass
            res.append(one[0] if one else ([], "ERROR_PARSE"))

    lines = "".join(json.dumps(
        {"uid": u["uid"], "chunk": idx, "model": args.tag,
         "labels": r[0], "exclusion": r[1]}, ensure_ascii=False) + "\n"
        for u, r in zip(units, res))
    with _lock:
        fh.write(lines); fh.flush()
        _stat["done"] += 1; _stat["units"] += len(units)
        _stat["fail"] += sum(1 for r in res if r[1] == "ERROR_PARSE")
        d, s = _stat["done"], _stat
        if d % 20 == 0 or d == 1:
            el = time.time() - s["t0"]; rate = d / el
            eta = (args.total_chunks - d - args.skipped) / rate / 60 if rate else 0
            print(f"  chunk {d + args.skipped:>6,}/{args.total_chunks:,} "
                  f"| {rate*60:5.1f} chunk/mnt | ETA {eta:6.1f} mnt "
                  f"| repair {s['repair']} split {s['split']} fail {s['fail']} "
                  f"| tok in {s['tin']/1e6:.1f}M out {s['tout']/1e6:.1f}M",
                  flush=True)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="nama pendek model, jadi kolom identitas")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="API_KEY")
    ap.add_argument("--work", default="work.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--json-mode", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="pilot: hanya N unit pertama")
    args = ap.parse_args()
    args.out = args.out or f"ann_{args.tag}.jsonl"

    key = os.environ.get(args.api_key_env)
    if not key:
        sys.exit(f"env {args.api_key_env} kosong")

    units = [json.loads(l) for l in open(args.work, encoding="utf-8")]
    if args.limit:
        units = units[:args.limit]
    chunks = [units[i:i + args.chunk] for i in range(0, len(units), args.chunk)]
    args.total_chunks = len(chunks)

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8"):
            try: done.add(json.loads(l)["chunk"])
            except Exception: pass
    todo = [(i, c) for i, c in enumerate(chunks) if i not in done]
    args.skipped = len(chunks) - len(todo)

    print(f"[{args.tag}] {len(units):,} unit -> {len(chunks):,} chunk @ {args.chunk}"
          f" | sudah selesai {args.skipped:,} | sisa {len(todo):,}"
          f" | concurrency {args.concurrency}", flush=True)
    if not todo:
        print("  sudah lengkap."); return

    with open(args.out, "a", encoding="utf-8") as fh, \
         ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(lambda t: do_chunk(t[0], t[1], args, key, fh), todo))

    el = (time.time() - _stat["t0"]) / 60
    print(f"[{args.tag}] SELESAI {_stat['units']:,} unit dalam {el:.1f} menit"
          f" | repair {_stat['repair']} split {_stat['split']} fail {_stat['fail']}"
          f" | token in {_stat['tin']/1e6:.1f}M out {_stat['tout']/1e6:.1f}M", flush=True)


if __name__ == "__main__":
    main()
