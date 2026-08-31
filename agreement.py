#!/usr/bin/env python3
"""Skor agreement multi-LLM per-SLOT + antrian adjudikasi bertingkat.

Unit skor adalah SLOT, bukan review: tiap unit menghasilkan 8 slot (satu per
kategori aspek), tiap slot bernilai 4-arah {ABSENT, POSITIVE, NEGATIVE, NEUTRAL}.
Mengikuti Wittlinger et al. (medRxiv 2026) yang menskor per-variabel (910 laporan
x 5 variabel), bukan per-laporan. Exact-match tingkat review dengan 8 aspek akan
menandai hampir semua review sebagai "disagree" dan angkanya tidak berguna.

Melaporkan raw agreement DAN Fleiss' kappa, keseluruhan dan per-aspek, serta
terpisah untuk subset ber-aspek -- karena mayoritas slot akan unanimous ABSENT
dan itu menggelembungkan raw agreement (paradoks kappa; Artstein & Poesio).

  python3 agreement.py ann_A.jsonl ann_B.jsonl ann_C.jsonl
"""
import json, sys, collections
from ontology import ASPECTS, ABSENT

CATS = [ABSENT, "POSITIVE", "NEGATIVE", "NEUTRAL"]


def load(path):
    d = {}
    for l in open(path, encoding="utf-8"):
        try: r = json.loads(l)
        except Exception: continue
        slot = {a: ABSENT for a in ASPECTS}
        for lb in r.get("labels") or []:
            if lb["aspect"] in slot:
                slot[lb["aspect"]] = lb["polarity"]
        d[r["uid"]] = (slot, r.get("exclusion"))
    return d


def fleiss(counts):
    """counts: list of dict kategori->jumlah rater. Semua item n rater sama."""
    counts = [c for c in counts if sum(c.values()) > 1]
    if not counts: return float("nan")
    N = len(counts); n = sum(counts[0].values())
    P = [(sum(v * v for v in c.values()) - n) / (n * (n - 1)) for c in counts]
    tot = collections.Counter()
    for c in counts: tot.update(c)
    pe = sum((tot[k] / (N * n)) ** 2 for k in tot)
    pb = sum(P) / N
    return (pb - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    paths = sys.argv[1:]
    if len(paths) < 3:
        sys.exit("butuh >=3 file anotasi (satu per model)")
    anns = [load(p) for p in paths]
    tags = [p.split("/")[-1].replace("ann_", "").replace(".jsonl", "") for p in paths]
    N = len(anns)                       # jumlah rater; laporan menyesuaikan
    MAJ = N // 2 + 1                    # ambang mayoritas jelas
    uids = sorted(set.intersection(*(set(a) for a in anns)))
    print(f"model            : {', '.join(tags)}")
    print(f"rater            : {N}")
    print(f"unit beririsan   : {len(uids):,}  (per file: "
          f"{', '.join(f'{len(a):,}' for a in anns)})")
    print(f"slot keputusan   : {len(uids)*len(ASPECTS):,}\n")

    per_aspect = {a: {"cnt": collections.Counter(), "fk": []} for a in ASPECTS}
    # §11 mewajibkan DUA agreement terpisah:
    #   (a) DETEKSI aspek  : hadir vs tidak hadir  -> kualitas ACD
    #   (b) PASANGAN       : aspek + sentimen      -> kualitas ACD+ACSA
    det_score, det_fk = collections.Counter(), []
    pair_on_detected = collections.Counter()   # polaritas, hanya saat deteksi bulat
    slot_score = collections.Counter()
    unit_tier, aspected = {}, 0
    fk_all, fk_asp = [], []

    for uid in uids:
        vals = [a[uid][0] for a in anns]
        has_aspect = any(v[x] != ABSENT for v in vals for x in ASPECTS)
        if has_aspect: aspected += 1
        worst = N
        for a in ASPECTS:
            c = collections.Counter(v[a] for v in vals)
            top = c.most_common(1)[0][1]           # 3 = bulat, 2 = mayoritas, 1 = beda semua
            slot_score[top] += 1

            # (a) deteksi: ABSENT vs ADA, mengabaikan polaritas
            d = collections.Counter("ABSENT" if v[a] == ABSENT else "PRESENT" for v in vals)
            det_score[d.most_common(1)[0][1]] += 1
            det_fk.append(dict(d))
            # (b) polaritas, dihitung HANYA bila ketiganya sepakat aspek itu hadir
            if d.get("PRESENT") == N:
                pc = collections.Counter(v[a] for v in vals)
                pair_on_detected[pc.most_common(1)[0][1]] += 1
            per_aspect[a]["cnt"][top] += 1
            per_aspect[a]["fk"].append(dict(c))
            fk_all.append(dict(c))
            if not (len(c) == 1 and ABSENT in c):
                fk_asp.append(dict(c))
            worst = min(worst, top)
        unit_tier[uid] = worst

    def _lab(sc):
        if sc == N:   return f"{sc}/{N} bulat"
        if sc >= MAJ: return f"{sc}-{N-sc} mayoritas"
        if sc == 1:   return "semua beda"
        return f"{sc}-... tanpa mayoritas"

    tot = sum(slot_score.values())
    print("=== DISTRIBUSI SKOR AGREEMENT PER-SLOT ===")
    for sc in range(N, 0, -1):
        lab = _lab(sc)
        print(f"  {lab:20s} {slot_score[sc]:9,}  {100*slot_score[sc]/tot:6.2f}%")
    print(f"\n  raw agreement (bulat)      : {100*slot_score[N]/tot:.2f}%")
    print(f"  Fleiss kappa  semua slot   : {fleiss(fk_all):.4f}")
    print(f"  Fleiss kappa  slot ber-aspek: {fleiss(fk_asp):.4f}   "
          f"(n={len(fk_asp):,}) <- angka yang jujur")

    print("\n=== (a) AGREEMENT DETEKSI ASPEK (ACD) — hadir vs tidak, abaikan polaritas ===")
    dt = sum(det_score.values())
    for sc in range(N, 0, -1):
        lab = _lab(sc)
        print(f"  {lab:20s} {det_score[sc]:9,}  {100*det_score[sc]/dt:6.2f}%")
    print(f"  Fleiss kappa deteksi       : {fleiss(det_fk):.4f}")

    print("\n=== (b) AGREEMENT POLARITAS pada aspek yang deteksinya BULAT (ACSA) ===")
    pt = sum(pair_on_detected.values())
    if pt:
        for sc in range(N, 0, -1):
            lab = _lab(sc)
            print(f"  {lab:20s} {pair_on_detected[sc]:9,}  {100*pair_on_detected[sc]/pt:6.2f}%")
        print(f"  (n={pt:,} slot; memisahkan error ACSA dari error ACD)")
    else:
        print("  (tidak ada slot dengan deteksi bulat)")

    print("\n=== (c) PER ASPEK — pasangan aspek+sentimen ===")
    print(f"  {'aspek':30s} {'bulat':>10s} {'mayoritas':>11s} {'tanpa maj.':>11s} "
          f"{'bulat%':>8s} {'kappa':>7s}")
    for a in ASPECTS:
        c = per_aspect[a]["cnt"]; t = sum(c.values())
        maj = sum(c[x] for x in range(MAJ, N))
        nom = sum(c[x] for x in range(1, MAJ))
        print(f"  {a:30s} {c[N]:10,} {maj:11,} {nom:11,} "
              f"{100*c[N]/t:7.2f}% {fleiss(per_aspect[a]['fk']):7.4f}")

    tc = collections.Counter(unit_tier.values())
    print(f"\n=== ANTRIAN ADJUDIKASI (unit, bukan slot) ===")
    print(f"  unit ber-aspek (>=1 model)          : {aspected:,}")
    t1 = tc[N]
    t2 = sum(tc[x] for x in range(MAJ, N))
    t3 = sum(tc[x] for x in range(1, MAJ))
    print(f"  TIER 1  semua slot bulat      -> terima : {t1:8,}  {100*t1/len(uids):5.2f}%")
    print(f"  TIER 2  mayoritas jelas       -> vote   : {t2:8,}  {100*t2/len(uids):5.2f}%")
    print(f"  TIER 3  tanpa mayoritas jelas -> MANUSIA: {t3:8,}  {100*t3/len(uids):5.2f}%")
    for m in (0.5, 1, 3):
        h = t3 * m / 60
        print(f"      tier 3 @ {m:>3} mnt/unit : {h:8,.0f} jam = {h/8:6,.0f} hari kerja")

    for lo, hi, name in ((1, MAJ, "tier3_manusia"), (MAJ, N, "tier2_vote")):
        with open(f"adjudicate_{name}.jsonl", "w", encoding="utf-8") as f:
            for uid in uids:
                if not (lo <= unit_tier[uid] < hi): continue
                f.write(json.dumps({"uid": uid, "min_agreement": unit_tier[uid],
                                    "n_raters": N, **{
                    t: {"labels": [{"aspect": k, "polarity": v}
                                   for k, v in anns[i][uid][0].items() if v != ABSENT],
                        "exclusion": anns[i][uid][1]}
                    for i, t in enumerate(tags)}}, ensure_ascii=False) + "\n")
        print(f"  -> adjudicate_{name}.jsonl")


if __name__ == "__main__":
    main()
