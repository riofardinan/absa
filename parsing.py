"""Parsing + validasi output LLM. Dipakai annotate_offline.py dan tools lain."""
import json, re
from ontology import ASPECTS, POLARITIES, EXCLUSIONS

VALID_A, VALID_P, VALID_E = set(ASPECTS), set(POLARITIES), set(EXCLUSIONS)

def _scan_objects(t):
    """Ambil setiap objek JSON top-level. Menangani array, JSON Lines, dan
    output terpotong (objek terakhir yang tidak lengkap dibuang)."""
    out, depth, start, instr, esc = [], 0, None, False, False
    for i, ch in enumerate(t):
        if instr:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try: out.append(json.loads(t[start:i + 1]))
                    except json.JSONDecodeError: pass
                    start = None
    return out


def extract_json(txt):
    t = re.sub(r"^\s*```(?:json)?|```\s*$", "", txt.strip(), flags=re.M).strip()
    i, k = t.find("["), t.rfind("]")
    if i != -1 and k > i:                       # array utuh: jalur cepat
        try:
            v = json.loads(t[i:k + 1])
            if isinstance(v, list): return v
        except json.JSONDecodeError:
            pass
    return _scan_objects(t)                     # JSONL / array rusak / terpotong


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

