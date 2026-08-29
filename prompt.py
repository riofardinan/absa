"""Pembangun prompt anotasi ACD+ACSA (mode batch-first untuk kecepatan)."""
import json
from ontology import ASPECTS, POLARITIES, ASPECT_DEF, BOUNDARIES

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
]


def _fewshot_block():
    out = []
    for i, (text, pairs, exc) in enumerate(FEWSHOT, 1):
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

Jangan memaksakan kategori yang tidak cocok. Lebih baik OUT_OF_ONTOLOGY atau INSUFFICIENT_CONTEXT daripada label salah.
Bila "labels" tidak kosong, "exclusion" HARUS null.

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

Keluarkan HANYA satu array JSON berisi TEPAT {n} objek, urut sesuai nomor di atas.
Setiap objek: {{"id": <nomor>, "labels": [{{"aspect": ..., "polarity": ...}}], "exclusion": <kode atau null>}}
Tanpa penjelasan, tanpa blok kode markdown."""


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
