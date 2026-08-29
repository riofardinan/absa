#!/usr/bin/env python3
"""Ukur limit & throughput NYATA sebuah endpoint dalam ~2 menit.

Jangan percaya angka limit di dokumentasi — ukur sendiri sebelum commit 5 jam.

  export API_KEY=...
  python3 probe.py --base-url <URL> --model <MODEL> --chunk 25 --concurrency 8
"""
import argparse, json, os, sys, time, threading, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from prompt import build_batch
from annotate import parse_chunk

lk = threading.Lock()
S = {"ok": 0, "429": 0, "err": 0, "bad": 0, "tin": 0, "tout": 0, "lat": []}


def one(texts, args, key):
    body = {"model": args.model, "temperature": 0, "max_tokens": args.max_tokens,
            "messages": [{"role": "user", "content": build_batch(texts)}]}
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            j = json.loads(r.read())
        lat = time.time() - t
        txt = j["choices"][0]["message"]["content"]
        u = j.get("usage") or {}
        ok = parse_chunk(txt, len(texts)) is not None
        with lk:
            S["ok"] += 1; S["lat"].append(lat)
            S["tin"] += u.get("prompt_tokens", 0); S["tout"] += u.get("completion_tokens", 0)
            if not ok: S["bad"] += 1
    except urllib.error.HTTPError as e:
        with lk:
            if e.code == 429:
                S["429"] += 1
                for h in ("Retry-After", "X-RateLimit-Limit-Requests",
                          "x-ratelimit-limit-requests", "X-RateLimit-Remaining-Requests"):
                    if e.headers.get(h): S.setdefault("hdr", {})[h] = e.headers.get(h)
            else:
                S["err"] += 1; S.setdefault("msg", str(e)[:200])
    except Exception as e:
        with lk: S["err"] += 1; S.setdefault("msg", str(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="API_KEY")
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--calls", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args()
    key = os.environ.get(a.api_key_env) or "sk-noauth"   # vLLM lokal tak butuh key

    units = [json.loads(l)["text"] for l in open("work.jsonl", encoding="utf-8")][:a.chunk * a.calls]
    batches = [units[i:i + a.chunk] for i in range(0, len(units), a.chunk)][:a.calls]
    print(f"probe: {len(batches)} call x {a.chunk} review, concurrency {a.concurrency}\n")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(lambda b: one(b, a, key), batches))
    el = time.time() - t0

    n = S["ok"]
    print(f"berhasil        : {n}/{len(batches)}   429: {S['429']}   error lain: {S['err']}")
    if S.get("msg"): print(f"pesan error     : {S['msg']}")
    if S.get("hdr"): print(f"header limit    : {S['hdr']}")
    if not n:
        sys.exit("\nsemua call gagal — cek base-url / model / API key.")
    S["lat"].sort()
    med = S["lat"][len(S["lat"]) // 2]
    rpm = n / el * 60
    print(f"latensi         : median {med:.1f}s  p90 {S['lat'][int(len(S['lat'])*.9)]:.1f}s")
    print(f"throughput      : {rpm:.1f} call/menit  ({rpm*a.chunk:,.0f} review/menit)")
    print(f"output invalid  : {S['bad']}/{n}  ({100*S['bad']/n:.1f}%)  <- >5% berarti turunkan --chunk")
    print(f"token/call      : in {S['tin']//n:,}  out {S['tout']//n:,}")

    U = 415480
    calls = -(-U // a.chunk)
    print(f"\n=== PROYEKSI 415.480 unit @ chunk-{a.chunk} ===")
    print(f"  call dibutuhkan : {calls:,} per model  ({3*calls:,} untuk 3 model)")
    print(f"  waktu 1 model   : {calls/rpm/60:.1f} jam   |  3 model paralel: ~{calls/rpm/60:.1f} jam")
    print(f"  token 1 model   : in {S['tin']//n*calls/1e6:.0f}M  out {S['tout']//n*calls/1e6:.0f}M")
    if S["429"]:
        print(f"\n  ADA 429 -> turunkan --concurrency, atau naikkan --chunk supaya "
              f"call berkurang.")
    print(f"\n  Cek RPD: kalau free tier membatasi request/hari, {calls:,} call/model "
          f"kemungkinan besar melampauinya.")


if __name__ == "__main__":
    main()
