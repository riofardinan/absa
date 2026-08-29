"""Bangun work list anotasi dari fintech_reviews_curated.csv.

Dua langkah, keduanya deterministik (tanpa LLM), sesuai tabel keputusan cleaning
pada dokumen riset:
  1. Exclusion by rule  : empty / emoji-only / malformed  -> tidak dikirim ke LLM
  2. Dedup by normalized key : teks identik dianotasi SEKALI, label dipropagasi
     balik ke semua reviewId anggota grup.

Output:
  work.jsonl   : {"uid", "text", "n_rows"}          -> yang dikirim ke LLM
  members.json : {uid: [reviewId, ...]}             -> untuk propagasi balik
  ruled.jsonl  : {"reviewId", "exclusion"}          -> hasil aturan, sudah final
"""
import csv, json, sys, collections

csv.field_size_limit(10**9)
SRC = sys.argv[1] if len(sys.argv) > 1 else "../fintech_reviews_curated.csv"

groups = collections.OrderedDict()   # normkey -> {"text":..., "ids":[...]}
ruled, n = [], 0

with open(SRC, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        n += 1
        rid = row["reviewId"]
        text = (row["content_clean"] or "").strip()

        if not text or row["flag_empty_content"] == "True":
            ruled.append({"reviewId": rid, "exclusion": "UNINTERPRETABLE",
                          "rule": "empty_content"}); continue
        if row["flag_emoji_symbol_only"] == "True":
            ruled.append({"reviewId": rid, "exclusion": "UNINTERPRETABLE",
                          "rule": "emoji_symbol_only"}); continue
        if row["flag_malformed_candidate"] == "True":
            ruled.append({"reviewId": rid, "exclusion": "UNINTERPRETABLE",
                          "rule": "malformed"}); continue

        key = (row["normalized_duplicate_key"] or text).strip()
        g = groups.get(key)
        if g is None:
            groups[key] = {"text": text, "ids": [rid]}
        else:
            g["ids"].append(rid)

with open("work.jsonl", "w", encoding="utf-8") as fw, \
     open("members.json", "w", encoding="utf-8") as fm, \
     open("ruled.jsonl", "w", encoding="utf-8") as fr:
    members = {}
    for i, g in enumerate(groups.values()):
        uid = f"u{i:07d}"
        members[uid] = g["ids"]
        fw.write(json.dumps({"uid": uid, "text": g["text"],
                             "n_rows": len(g["ids"])}, ensure_ascii=False) + "\n")
    json.dump(members, fm, ensure_ascii=False)
    for r in ruled:
        fr.write(json.dumps(r, ensure_ascii=False) + "\n")

u = len(groups)
print(f"baris CSV            : {n:,}")
print(f"dibuang oleh aturan  : {len(ruled):,}")
print(f"unit anotasi (unik)  : {u:,}   (hemat {n-len(ruled)-u:,} = "
      f"{100*(n-len(ruled)-u)/(n-len(ruled)):.1f}%)")
for cs in (25, 50):
    print(f"  chunk-{cs}: {-(-u//cs):,} call/model  |  {3*-(-u//cs):,} call untuk 3 model")
