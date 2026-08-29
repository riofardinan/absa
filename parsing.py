"""Parsing + validasi output LLM. Dipakai annotate_offline.py dan tools lain."""
import json, re
from ontology import ASPECTS, POLARITIES, EXCLUSIONS

VALID_A, VALID_P, VALID_E = set(ASPECTS), set(POLARITIES), set(EXCLUSIONS)

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

