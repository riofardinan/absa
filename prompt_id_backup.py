"""Pembangun prompt anotasi ACD+ACSA (mode batch-first untuk kecepatan)."""
import json
from ontology import (ASPECTS, POLARITIES, ASPECT_DEF, BOUNDARIES,
                      ASPECT_CODE, POLARITY_CODE, EXCLUSION_CODE)

# Few-shot diambil langsung dari tabel "Aturan keputusan" + contoh IC/EC dokumen,
# dipetakan ke ontologi 8-aspek.
FEWSHOT = [
    ("Transfernya gagal.", [("TRANSACTION_PERFORMANCE", "NEGATIVE")], None),
    ("Transfer gagal karena server error.",
     [("TRANSACTION_PERFORMANCE", "NEGATIVE"), ("APP_TECHNICAL_PERFORMANCE", "NEGATIVE")], None),
    ("Tidak bisa login setelah update.",
     [("ACCOUNT_ACCESS_REGISTRATION", "NEGATIVE"), ("APP_TECHNICAL_PERFORMANCE", "NEGATIVE")], None),
    ("Menu transfer sulit ditemukan.",
     [("UI_UX", "NEGATIVE"), ("FEATURE_FUNCTIONALITY", "NEUTRAL")], None),
    ("CS cepat menyelesaikan masalah saldo saya.",
     [("CUSTOMER_SERVICE", "POSITIVE"), ("TRANSACTION_PERFORMANCE", "NEUTRAL")], None),
    ("Tampilannya bagus tetapi sering crash.",
     [("UI_UX", "POSITIVE"), ("APP_TECHNICAL_PERFORMANCE", "NEGATIVE")], None),
    ("transfer cepat tapi biaya mahal",          # IC4 multi-aspek
     [("TRANSACTION_PERFORMANCE", "POSITIVE"), ("FEES_CHARGES", "NEGATIVE")], None),
    ("dari tadi gak bisa masuk",                  # IC5 aspek implisit
     [("ACCOUNT_ACCESS_REGISTRATION", "NEGATIVE")], None),
    ("tf g masuk2",                               # IC7 informal/singkatan
     [("TRANSACTION_PERFORMANCE", "NEGATIVE")], None),
    ("Saya top up melalui BCA.",                  # IC8 faktual -> NEUTRAL (EC6 nonaktif)
     [("TRANSACTION_PERFORMANCE", "NEUTRAL")], None),
    ("Aplikasinya bagus.", [], "NO_SPECIFIC_ASPECT"),          # EC2
    ("Mobile Legends sekarang makin bagus", [], "OUT_OF_SCOPE"),  # EC1
    ("gagal terus", [], "INSUFFICIENT_CONTEXT"),               # EC4
    ("ajsdh %% 123 ???", [], "UNINTERPRETABLE"),               # EC5
    # ABSTAIN: aspek yang SAMA positif+negatif tanpa dominan (§5)
    ("transfernya kadang cepat kadang gagal",
     [("TRANSACTION_PERFORMANCE", "ABSTAIN")], None),
    # SP: flag, bukan exclusion -> aspek tetap dilabeli (§7 EC7)
    ("pakai kode referral ABC123 dapat saldo gratis, fiturnya lengkap",
     [("FEATURE_FUNCTIONALITY", "POSITIVE")], None, "SP"),
]


def _fewshot_block(compact=True):
    out = []
    for i, item in enumerate(FEWSHOT, 1):
        text, pairs, exc = item[0], item[1], item[2]
        flag = item[3] if len(item) > 3 else None
        if compact:
            r = (f"-{EXCLUSION_CODE[exc]}" if exc else
                 " ".join(f"{ASPECT_CODE[a]}:{POLARITY_CODE[p]}" for a, p in pairs))
            if flag: r += f" {flag}"
            out.append(f'{i}. "{text}"\n   -> {i} {r}')
        else:
            obj = {"id": i,
                   "labels": [{"aspect": a, "polarity": p} for a, p in pairs],
                   "exclusion": exc}
            out.append(f'{i}. "{text}"\n   -> {json.dumps(obj, ensure_ascii=False)}')
    return "\n".join(out)


def _rules():
    asp = "\n".join(f"- {a}: {ASPECT_DEF[a]}" for a in ASPECTS)
    bnd = "\n".join(f"- {b}" for b in BOUNDARIES)
    return f"""Kamu adalah anotator Aspect-Based Sentiment Analysis untuk ulasan aplikasi dompet digital (e-wallet) Indonesia: DANA, GoPay, OVO, ShopeePay.

Tugas: Aspect Category Detection (ACD) + Aspect Category Sentiment Analysis (ACSA).
Untuk setiap review, tentukan kategori aspek mana saja yang dibahas, dan polaritas sentimen terhadap MASING-MASING aspek itu.

=== 8 KATEGORI ASPEK ===
{asp}

=== POLARITAS ===
{", ".join(POLARITIES)}
- POSITIVE: pengguna memuji / puas terhadap aspek itu.
- NEGATIVE: pengguna mengeluh / tidak puas terhadap aspek itu.
- NEUTRAL: aspek dibahas secara faktual/deskriptif tanpa evaluasi positif atau negatif yang jelas.

=== BATAS ANTAR-ASPEK (sering tertukar) ===
{bnd}

=== ATURAN INKLUSI ===
- Satu review BOLEH memiliki lebih dari satu pasangan (aspek, polaritas).
- Aspek IMPLISIT diperbolehkan selama kategorinya jelas dari konteks; aspek tidak harus muncul sebagai kata literal.
- Review pendek tetap dilabeli jika informasi aspeknya jelas ("transfer gagal" -> valid).
- Bahasa informal, typo, singkatan, slang, campur kode tetap dilabeli selama maknanya dapat dipahami.
- Menyebut aspek tanpa evaluasi jelas -> polaritas NEUTRAL.

=== ATURAN EKSKLUSI ===
Bila review tidak dapat dilabeli: "labels": [] dan isi "exclusion" dengan SATU kode:
- OUT_OF_SCOPE: tidak membahas e-wallet target / layanan / fitur / pengalaman penggunaannya.
- NO_SPECIFIC_ASPECT: ada opini tetapi tidak ada satu pun dari 8 kategori yang teridentifikasi ("mantap", "bagus banget").
- OUT_OF_ONTOLOGY: ada aspek jelas, tetapi tidak dapat dipetakan ke 8 kategori di atas.
- INSUFFICIENT_CONTEXT: konteks terlalu ambigu untuk menentukan kategori secara andal.
- UNINTERPRETABLE: teks tidak dapat diinterpretasikan secara semantik, bahkan setelah mempertimbangkan typo/slang/singkatan/campur kode.
- SPAM_FAKE: jelas promosi/spam, tidak merepresentasikan pengalaman pengguna.

Jangan memaksakan kategori yang tidak cocok. Lebih baik OO (out of ontology) atau IC (insufficient context) daripada label salah.

=== FORMAT OUTPUT ===
SATU BARIS per review. Tanpa penjelasan, tanpa JSON, tanpa markdown.

  <nomor> <ASPEK>:<POLARITAS> <ASPEK>:<POLARITAS> ...
  <nomor> -<KODE_EKSKLUSI>                        (bila tidak dapat dilabeli)

Kode aspek:
  SEC = Security                    ACC = Account Access & Registration
  FEA = Feature & Functionality     UIX = UI / UX
  CSV = Customer Service            FEE = Fees & Charges
  TRX = Transaction Performance     APP = App / Technical Performance
Kode polaritas:  P = Positive   N = Negative   U = Neutral
                 X = ABSTAIN, HANYA bila aspek yang SAMA membawa evaluasi positif
                     dan negatif sekaligus tanpa ada yang jelas dominan
Kode eksklusi :  OS = out of scope   NS = no specific aspect   OO = out of ontology
                 IC = insufficient context   UN = uninterpretable
Flag tambahan :  SP = konten promosi/spam. Ini FLAG, BUKAN alasan pembuangan.
                 Tetap labeli aspeknya bila bisa; tulis SP di akhir baris.

Contoh baris:   3 TRX:N APP:N        7 -NS        9 TRX:X        12 FEA:P SP

=== CONTOH ===
{_fewshot_block()}"""


def build_batch(texts):
    """N review per call. Wajib mengembalikan tepat len(texts) objek, urut, ber-id."""
    n = len(texts)
    numbered = "\n".join(f'{i}. "{t}"' for i, t in enumerate(texts, 1))
    return f"""{_rules()}

=== ANOTASI SEKARANG ===
Di bawah ini ada {n} review. Labeli SETIAP review secara INDEPENDEN.
Jangan biarkan isi satu review mempengaruhi label review lain.

{numbered}

Keluarkan TEPAT {n} baris, urut dari 1 sampai {n}, satu baris per review.
Tanpa penjelasan, tanpa JSON, tanpa markdown."""


def build_single(text):
    return build_batch([text])


if __name__ == "__main__":
    import sys
    r = _rules()
    print(f"prefix rules+fewshot : {len(r):6d} char  (~{len(r)/3.5:.0f} token est.)")
    for n in (1, 25, 50):
        b = build_batch(["saldo kepotong tapi transaksi gagal"] * n)
        est = len(b) / 3.5
        print(f"prompt batch-{n:<3d}     : {len(b):6d} char  (~{est:.0f} token est.)"
              f"  -> {est/n:6.0f} token/review")
    if "--show" in sys.argv:
        print("\n" + build_batch(["tf g masuk2", "mantap", "cs nya lelet banget"]))
