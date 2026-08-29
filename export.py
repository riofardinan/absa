#!/usr/bin/env python3
"""Ekspor hasil anotasi ke format spec §9 (JSONL per review) dan §10 (long CSV).

Menggabungkan ann_<tag>.jsonl (per unit dedup) + members.json (uid -> reviewId)
+ CSV asli (app, content), sehingga tiap reviewId punya satu record.
Record excluded/abstain TETAP disimpan (§9).

  python3 export.py --tag qwen --csv ../fintech_reviews_curated.csv
"""
import argparse, csv, json, os
from ontology import ASPECT_DISPLAY, POLARITY_DISPLAY, ABSTAIN, ABSTAIN_REASON

csv.field_size_limit(10**9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--csv", default="../fintech_reviews_curated.csv")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--members", default="members.json")
    ap.add_argument("--ruled", default="ruled.jsonl")
    a = ap.parse_args()
    ann_path = a.ann or f"ann_{a.tag}.jsonl"

    members = json.load(open(a.members, encoding="utf-8"))
    ann = {}
    for l in open(ann_path, encoding="utf-8"):
        r = json.loads(l); ann[r["uid"]] = r
    rid2uid = {rid: uid for uid, ids in members.items() for rid in ids}
    ruled = {json.loads(l)["reviewId"]: json.loads(l)
             for l in open(a.ruled, encoding="utf-8")} if os.path.exists(a.ruled) else {}

    out_j = f"annotations_{a.tag}.jsonl"
    out_l = f"long_{a.tag}.csv"
    n = n_ann = n_exc = n_abst = n_spam = n_err = n_pair = 0

    with open(a.csv, encoding="utf-8-sig", newline="") as f, \
         open(out_j, "w", encoding="utf-8") as fj, \
         open(out_l, "w", encoding="utf-8", newline="") as fl:
        w = csv.writer(fl)
        w.writerow(["review_id", "app", "aspect", "sentiment",
                    "annotation_method", "model_name", "prompt_version"])
        for row in csv.DictReader(f):
            n += 1
            rid, app, text = row["reviewId"], row["app"], row["content"]

            if rid in ruled:                       # dibuang aturan deterministik
                rec = {"review_id": rid, "app": app, "text": text,
                       "eligible": False, "exclusion_reason": ruled[rid]["exclusion"],
                       "status": "EXCLUDED", "annotations": [], "flags": [],
                       "method": "RULE", "provider": None, "model_name": None,
                       "prompt_version": None, "parser_valid": True,
                       "rule": ruled[rid]["rule"]}
                n_exc += 1
                fj.write(json.dumps(rec, ensure_ascii=False) + "\n"); continue

            r = ann.get(rid2uid.get(rid))
            if r is None:                          # belum dianotasi
                fj.write(json.dumps({"review_id": rid, "app": app, "text": text,
                                     "eligible": None, "exclusion_reason": None,
                                     "status": "PENDING", "annotations": [], "flags": [],
                                     "method": None, "provider": None, "model_name": None,
                                     "prompt_version": None, "parser_valid": None},
                                    ensure_ascii=False) + "\n")
                continue

            labels = r.get("labels") or []
            exc = r.get("exclusion")
            has_abstain = any(l["polarity"] == ABSTAIN for l in labels)
            if not r.get("parser_valid", True):
                status = "PARSE_ERROR"; n_err += 1
            elif labels:
                status = "ABSTAIN" if has_abstain else "ANNOTATED"; n_ann += 1
                n_abst += has_abstain
            else:
                status = "EXCLUDED"; n_exc += 1

            anns = [{"aspect": ASPECT_DISPLAY[l["aspect"]],
                     "sentiment": POLARITY_DISPLAY[l["polarity"]],
                     **({"reason": ABSTAIN_REASON} if l["polarity"] == ABSTAIN else {})}
                    for l in labels]
            flags = r.get("flags") or []
            n_spam += bool(flags)

            fj.write(json.dumps({
                "review_id": rid, "app": app, "text": text,
                "eligible": bool(labels), "exclusion_reason": exc,
                "status": status, "annotations": anns, "flags": flags,
                "method": r.get("method", "LLM"), "provider": r.get("provider"),
                "model_name": r.get("model_name"),
                "prompt_version": r.get("prompt_version"),
                "parser_valid": r.get("parser_valid", True),
                "annotated_at": r.get("annotated_at"),
            }, ensure_ascii=False) + "\n")

            for x in anns:                         # §10 long: 1 baris per pasangan
                w.writerow([rid, app, x["aspect"], x["sentiment"], "LLM",
                            r.get("model_name"), r.get("prompt_version")])
                n_pair += 1

    print(f"baris CSV        : {n:,}")
    print(f"ANNOTATED        : {n_ann:,}   (di antaranya ABSTAIN: {n_abst:,})")
    print(f"EXCLUDED         : {n_exc:,}")
    print(f"PARSE_ERROR      : {n_err:,}")
    print(f"ber-flag spam    : {n_spam:,}")
    print(f"pasangan (long)  : {n_pair:,}")
    print(f"\n-> {out_j}  (§9)\n-> {out_l}  (§10)")


if __name__ == "__main__":
    main()
