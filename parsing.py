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
        if a in VALID_A and p in VALID_P | {"ABSTAIN"} and a not in seen:
            seen.add(a); labels.append({"aspect": a, "polarity": p})
    exc = obj.get("exclusion")
    exc = str(exc).strip().upper() if exc not in (None, "", "null", "NULL") else None
    if exc is not None and exc not in VALID_E:
        exc = "OUT_OF_ONTOLOGY"
    if labels:
        exc = None                      # aturan: labels terisi -> exclusion null
    elif exc is None:
        return None                     # kosong tanpa alasan -> anggap gagal
    return labels, exc, []

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



# ------------------------------------------------------ format ringkas (wire)
from ontology import (CODE_ASPECT, CODE_POLARITY, CODE_EXCLUSION, SPAM_FLAG,
                      ASPECTS, EXCLUSIONS)

_LINE = re.compile(r"^\s*(\d+)[\.\):]?\s+(.*?)\s*$")
_PAIR = re.compile(r"^([A-Z_]{2,32}):([PNUX])$")     # X = ABSTAIN (§5)
_EXC  = re.compile(r"^-?([A-Z_]{2,24})$")
# Model kadang memakai nama panjang (CUSTOMER_SERVICE:N) alih-alih sandi (CSV:N),
# dan kadang menulis eksklusi tanpa tanda minus (IC, bukan -IC). Terima keduanya.
_ASP_ANY = dict(CODE_ASPECT, **{a: a for a in ASPECTS})
_EXC_ANY = dict(CODE_EXCLUSION, **{e: e for e in EXCLUSIONS})


def _compact_map(txt, size):
    """Peta id -> (labels, exclusion, flags) untuk SETIAP baris yang valid.
    Baris cacat dilewati, tidak membatalkan baris lain."""
    got = {}
    for raw in txt.replace("```", "\n").split("\n"):
        m = _LINE.match(raw)
        if not m:
            continue
        i = int(m.group(1))
        if not 1 <= i <= size or i in got:
            continue
        body = m.group(2).strip()
        if not body:
            continue
        labels, seen, exc, bad, spam = [], set(), None, False, False
        for t in body.split():
            t = t.strip(",;").upper()
            p = _PAIR.match(t)
            if p and p.group(1) in _ASP_ANY:
                a = _ASP_ANY[p.group(1)]
                if a not in seen:
                    seen.add(a)
                    labels.append({"aspect": a, "polarity": CODE_POLARITY[p.group(2)]})
                continue
            if t in ("SP", "-SP"):              # flag, bukan exclusion (§7 EC7)
                spam = True
                continue
            e = _EXC.match(t)
            if e and e.group(1) in _EXC_ANY:
                exc = _EXC_ANY[e.group(1)]
                continue
            bad = True                          # token tak dikenal -> jangan tebak
        if bad and not labels and exc is None:
            continue
        if labels:
            exc = None                          # aturan: labels terisi -> exclusion null
        elif exc is None:
            if spam:
                exc = "NO_SPECIFIC_ASPECT"      # SP sendirian: tetap bukan alasan buang
            else:
                continue
        got[i] = (labels, exc, [SPAM_FLAG] if spam else [])
    return got


def parse_chunk_compact(txt, size):
    """All-or-nothing; dipakai saat butuh kepastian penuh."""
    got = _compact_map(txt, size)
    if len(got) != size:
        return None
    return [got[i] for i in range(1, size + 1)]


def parse_chunk_partial(txt, size, compact=True):
    """TIDAK all-or-nothing: satu baris cacat di antara 25 baris bagus tidak
    boleh membatalkan 25 unit. Kembalikan (hasil, id_yang_gagal); hasil[i] None
    untuk yang gagal sehingga hanya unit itu yang perlu diulang."""
    if compact:
        got = _compact_map(txt, size)
    else:
        got = {}
        for k, o in enumerate(extract_json(txt) or [], 1):
            if k > size:
                break
            r = norm_item(o)
            if r is not None:
                got[k] = r
    res = [got.get(i) for i in range(1, size + 1)]
    return res, [i for i in range(1, size + 1) if got.get(i) is None]
