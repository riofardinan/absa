#!/usr/bin/env python3
"""Bandingkan kandidat model dari hasil pilot, untuk memilih 3 terbaik.

Dua metrik yang menentukan, keduanya bisa diukur tanpa gold set:
  (1) fail %       -> kepatuhan format. >1% berarti model tidak andal.
  (2) over-label % -> kalibrasi "tidak ada aspek", pada subset kontrol
      (<=3 token DAN nol seed keyword). Spec §7 EC2 mendefinisikan kelas ini
      sebagai NO_SPECIFIC_ASPECT, dengan "mantap" sebagai contoh harfiah.
      Ini masalah kualitas terbesar yang terukur pada run A100 (25-81%).

  python3 compare_models.py ann_qwen35.jsonl ann_gemma4.jsonl ann_phi4.jsonl
"""
import json, re, sys, collections

SEEDS = re.compile(
    r'(?<![a-z])(keamanan|aman|penipuan|scam|fraud|akun|login|logout|otp|verifikasi|'
    r'password|fitur|menu|paylater|pinjaman|qris|layanan|tampilan|desain|navigasi|tombol|'
    r'customer service|cs|keluhan|respon|biaya|admin|fee|potongan|terpotong|transfer|top up|'
    r'pembayaran|transaksi|kirim|pending|gagal|berhasil|lambat|error|crash|lemot|lelet|lag|'
    r'loading|maintenance|bug)(?![a-z])')


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    ctrl, text = set(), {}
    for l in open("work.jsonl", encoding="utf-8"):
        r = json.loads(l)
        t = r["text"].strip().lower()
        text[r["uid"]] = t
        if len(t.split()) <= 3 and not SEEDS.search(t):
            ctrl.add(r["uid"])

    print(f"subset kontrol di korpus: {len(ctrl):,} unit\n")
    print(f"  {'model':<16}{'unit':>8}{'fail%':>8}{'kontrol':>9}{'over-label%':>13}"
          f"{'ABSTAIN':>9}{'spam':>7}{'aspek/unit':>12}")
    print("  " + "-" * 82)
    rows = []
    for p in paths:
        n = nf = nc = nover = nab = nsp = nlab = 0
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            n += 1
            if not r.get("parser_valid", True):
                nf += 1; continue
            labs = r.get("labels") or []
            nlab += len(labs)
            nab += any(x["polarity"] == "ABSTAIN" for x in labs)
            nsp += bool(r.get("flags"))
            if r["uid"] in ctrl:
                nc += 1
                nover += bool(labs)
        tag = p.replace("ann_", "").replace(".jsonl", "")[:15]
        ov = 100 * nover / nc if nc else float("nan")
        rows.append((tag, ov, 100 * nf / n))
        print(f"  {tag:<16}{n:>8,}{100*nf/n:>7.2f}%{nc:>9,}{ov:>12.1f}%"
              f"{nab:>9,}{nsp:>7,}{nlab/max(n-nf,1):>12.2f}")

    print("\n  CARA MEMILIH:")
    print("    fail%       < 1%   -> kalau lebih, model tidak andal, jangan dipakai")
    print("    over-label% lebih rendah = kalibrasi 'tidak ada aspek' lebih baik")
    print("               (run A100: qwen 25,0% · mistral 61,7% · llama 80,7%)")
    print("    WAJIB: 3 model dari KELUARGA PRETRAINING BERBEDA, bukan 3 terbaik")
    print("           -- error model sekeluarga berkorelasi (Wittlinger et al.)")
    ok = [r for r in rows if r[2] < 1.0]
    if ok:
        print(f"\n  lolos fail%<1: {', '.join(t for t,_,_ in sorted(ok,key=lambda x:x[1]))}"
              f"  (urut over-label terendah)")


if __name__ == "__main__":
    main()
