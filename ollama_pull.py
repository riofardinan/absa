#!/usr/bin/env python3
"""Unduh model lewat API Ollama — tanpa perlu CLI `ollama` di PATH.

  python3 ollama_pull.py qwen3.5:4b gemma4:e2b granite4:3b phi4-mini llama3.2:3b
"""
import json, sys, time, urllib.error, urllib.request

HOST = "http://localhost:11434"


def api(path, body=None, stream=False, timeout=30):
    req = urllib.request.Request(
        HOST + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("-")]
    for i, a in enumerate(sys.argv[1:]):
        if a == "--host":
            globals()["HOST"] = sys.argv[i + 2]; names.remove(sys.argv[i + 2])
    if not names:
        sys.exit(__doc__)

    try:
        v = json.load(api("/api/version", timeout=10))
        print(f"server Ollama: v{v.get('version','?')} @ {HOST}\n")
    except Exception as e:
        sys.exit(f"!! tidak bisa menghubungi {HOST}: {type(e).__name__}\n"
                 f"   Pastikan Ollama berjalan (ikon di system tray Windows),\n"
                 f"   atau jalankan `ollama serve`.")

    have = {m["name"] for m in json.load(api("/api/tags", timeout=20)).get("models", [])}
    for n in names:
        if n in have or f"{n}:latest" in have:
            print(f"  {n:18s} sudah ada, dilewati"); continue
        print(f"  {n:18s} mengunduh...", end="", flush=True)
        t0, last = time.time(), ""
        try:
            r = api("/api/pull", {"model": n, "stream": True}, timeout=7200)
            for line in r:
                if not line.strip():
                    continue
                d = json.loads(line)
                if "error" in d:
                    print(f"\r  {n:18s} GAGAL: {d['error'][:80]}"); break
                st = d.get("status", "")
                tot, done = d.get("total"), d.get("completed")
                if tot and done:
                    msg = f"{100*done/tot:5.1f}%  {done/2**30:5.2f}/{tot/2**30:.2f} GiB"
                else:
                    msg = st[:40]
                if msg != last:
                    print(f"\r  {n:18s} {msg:38s}", end="", flush=True); last = msg
            else:
                print(f"\r  {n:18s} SELESAI ({time.time()-t0:.0f} dtk){' '*20}")
        except urllib.error.HTTPError as e:
            print(f"\r  {n:18s} GAGAL HTTP {e.code}: {e.read()[:120].decode('utf-8','replace')}")
        except Exception as e:
            print(f"\r  {n:18s} GAGAL: {type(e).__name__}: {e}")

    print("\nmodel tersedia sekarang:")
    for m in json.load(api("/api/tags", timeout=20)).get("models", []):
        print(f"  {m['name']:28s} {(m.get('size') or 0)/2**30:6.2f} GiB")


if __name__ == "__main__":
    main()
