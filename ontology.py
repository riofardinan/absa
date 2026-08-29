"""Ontologi ABSA IFRAD: 8 aspect category + skema sentimen 3-arah.

Sumber label & kriteria: dokumen riset (tabel "Final aspek kita", "Aspek dan Kata
Kunci", IC1-IC8, EC1-EC7). Skema polaritas Positive/Negative/Neutral mengikuti
Pontiki et al. (2016) SemEval-2016 Task 5; 'conflict/Mixed' TIDAK dipakai karena
tidak ada pada SemEval-2016.

EC6 (NON_OPINION) berstatus NONAKTIF: dokumen menyatakan EC6 hanya berlaku bila
skema sentimen tanpa Neutral. Karena Neutral dipakai, mention faktual seperti
"saya top up lewat BCA" masuk sebagai NEUTRAL (lihat IC8).
"""

ASPECTS = [
    "SECURITY",
    "ACCOUNT_ACCESS_REGISTRATION",
    "FEATURE_FUNCTIONALITY",
    "UI_UX",
    "CUSTOMER_SERVICE",
    "FEES_CHARGES",
    "TRANSACTION_PERFORMANCE",
    "APP_TECHNICAL_PERFORMANCE",
]

POLARITIES = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

# Nilai slot ke-4 saat aspek tidak dibahas. Dipakai oleh agreement scorer:
# tiap review menghasilkan 8 slot, tiap slot 4-way {ABSENT, POS, NEG, NEU}.
ABSENT = "ABSENT"

# Exclusion code. EC6 sengaja dihilangkan (lihat docstring).
EXCLUSIONS = [
    "OUT_OF_SCOPE",          # EC1
    "NO_SPECIFIC_ASPECT",    # EC2
    "OUT_OF_ONTOLOGY",       # EC3
    "INSUFFICIENT_CONTEXT",  # EC4
    "UNINTERPRETABLE",       # EC5
    "SPAM_FAKE",             # EC7 - hanya bila high-confidence
]

ASPECT_DEF = {
    "SECURITY": (
        "Keamanan dana dan data: penipuan, scam, akun dibobol/dibajak, transaksi "
        "tidak sah, kebocoran data pribadi, kerugian finansial akibat kejahatan siber."),
    "ACCOUNT_ACCESS_REGISTRATION": (
        "Akses akun: login, logout, registrasi/daftar, OTP, PIN, verifikasi identitas, "
        "username, password, email, nomor HP, akun terblokir/dibekukan."),
    "FEATURE_FUNCTIONALITY": (
        "Ketersediaan, kelengkapan, dan berfungsinya fitur/menu/layanan: PayLater, "
        "pinjaman, QRIS sebagai fitur, voucher, promo sebagai fitur, fitur hilang "
        "setelah update."),
    "UI_UX": (
        "Tampilan dan kemudahan pakai: desain visual, tata letak, navigasi, scroll, "
        "tombol, ikon, kemudahan/kesulitan menemukan sesuatu, iklan yang mengganggu tampilan."),
    "CUSTOMER_SERVICE": (
        "Layanan pelanggan: respons CS, call center, pusat bantuan, aduan/komplain, "
        "tindak lanjut keluhan, chatbot bantuan."),
    "FEES_CHARGES": (
        "Biaya dan potongan: biaya admin, biaya transfer, fee, saldo terpotong sebagai "
        "biaya, bunga, denda, harga/tarif layanan."),
    "TRANSACTION_PERFORMANCE": (
        "Jalannya transaksi: transfer, top up, pembayaran, tagihan, QRIS sebagai "
        "transaksi, kirim uang, status pending/gagal/berhasil, dana tidak masuk, "
        "saldo terpotong tapi transaksi gagal, kecepatan proses transaksi."),
    "APP_TECHNICAL_PERFORMANCE": (
        "Kinerja teknis aplikasi: error, crash, force close, lemot/lelet, lag, loading "
        "lama, maintenance, gangguan server, koneksi, konsumsi memori, bug, gagal update."),
}

# Batas antar-aspek yang paling sering tertukar. Diturunkan dari tabel
# "Contoh guideline ringkas" (kolom "Tidak termasuk").
BOUNDARIES = [
    "Menu transfer sulit DITEMUKAN -> UI_UX, bukan FEATURE_FUNCTIONALITY.",
    "Transaksi GAGAL tanpa sebab teknis disebut -> TRANSACTION_PERFORMANCE saja, "
    "jangan tambahkan APP_TECHNICAL_PERFORMANCE.",
    "Aplikasi gagal dibuka karena server -> APP_TECHNICAL_PERFORMANCE, bukan SECURITY.",
    "Tidak bisa login -> ACCOUNT_ACCESS_REGISTRATION. Tambahkan SECURITY hanya bila ada "
    "indikasi penipuan/pembobolan/akun dibajak.",
    "Saldo terpotong sebagai BIAYA -> FEES_CHARGES. Saldo terpotong tapi transaksi GAGAL "
    "-> TRANSACTION_PERFORMANCE.",
    "Kegagalan transaksi tanpa membahas CS -> jangan beri CUSTOMER_SERVICE.",
    "Fitur yang tidak tersedia/hilang -> FEATURE_FUNCTIONALITY. Fitur ada tapi errornya "
    "teknis -> APP_TECHNICAL_PERFORMANCE.",
]
