"""Bangun work list anotasi dari fintech_reviews_curated.csv.

Sesuai spec §3: input anotasi adalah kolom `content` (teks asli), BUKAN
`content_clean` (berbeda pada 5.952 baris, mis. "kemana²" -> "kemana2").
`score` dan `replyContent` tidak pernah ikut.

Dedup memakai `content` PERSIS, bukan normalized key: anggota grup normalized
bisa punya teks berbeda, dan spec mewajibkan mempertahankan teks asli.

Exclusion by rule dibatasi pada kasus yang benar-benar tak terbantahkan
(spec §7 EC5: "bukan sekadar pendek/typo/slang"):
  - content kosong            -> UNINTERPRETABLE
  - emoji/simbol saja         -> NO_SPECIFIC_ASPECT   (bukan UNINTERPRETABLE:
                                 teksnya terbaca, hanya tidak ada aspek)
  - flag_malformed_candidate  -> TIDAK dibuang; itu heuristik, biar LLM yang menilai

Output:
  work.jsonl   {"uid","text","n_rows"}      -> dikirim ke LLM
  members.json {uid: [reviewId, ...]}       -> propagasi label balik
  ruled.jsonl  {"reviewId","exclusion","rule"} -> hasil aturan, final
"""
import csv, json, sys, collections

csv.field_size_limit(10**9)
SRC = sys.argv[1] if len(sys.argv) > 1 else "../fintech_reviews_curated.csv"

groups = collections.OrderedDict()
ruled, n = [], 0

with open(SRC, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        n += 1
        rid, text = row["reviewId"], (row["content"] or "").strip()

        if not text or row["flag_empty_content"] == "True":
            ruled.append({"reviewId": rid, "exclusion": "UNINTERPRETABLE",
                          "rule": "empty_content"}); continue
        if row["flag_emoji_symbol_only"] == "True":
            ruled.append({"reviewId": rid, "exclusion": "NO_SPECIFIC_ASPECT",
                          "rule": "emoji_symbol_only"}); continue

        g = groups.get(text)
        if g is None: groups[text] = [rid]
        else: g.append(rid)

with open("work.jsonl", "w", encoding="utf-8") as fw, \
     open("members.json", "w", encoding="utf-8") as fm, \
     open("ruled.jsonl", "w", encoding="utf-8") as fr:
    members = {}
    for i, (text, ids) in enumerate(groups.items()):
        uid = f"u{i:07d}"
        members[uid] = ids
        fw.write(json.dumps({"uid": uid, "text": text, "n_rows": len(ids)},
                            ensure_ascii=False) + "\n")
    json.dump(members, fm, ensure_ascii=False)
    for r in ruled:
        fr.write(json.dumps(r, ensure_ascii=False) + "\n")

u = len(groups)
print(f"baris CSV            : {n:,}")
print(f"dibuang oleh aturan  : {len(ruled):,}")
for k, v in collections.Counter(r["rule"] for r in ruled).items():
    print(f"    {k:22s} {v:,}")
print(f"unit anotasi (unik)  : {u:,}   (hemat {n-len(ruled)-u:,} = "
      f"{100*(n-len(ruled)-u)/(n-len(ruled)):.1f}%)")
for cs in (25, 50):
    print(f"  chunk-{cs}: {-(-u//cs):,} call/model  |  {3*-(-u//cs):,} call untuk 3 model")
