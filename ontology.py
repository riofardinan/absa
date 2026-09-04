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


# --------------------------------------------------------------- kode ringkas
# Format wire ringkas: waktu generasi didominasi token OUTPUT, dan nama panjang
# seperti TRANSACTION_PERFORMANCE + boilerplate JSON memakan ~100 tok/review.
# Sandi ini hanya dipakai di prompt & parsing; ann_*.jsonl tetap nama panjang.
ASPECT_CODE = {
    "SECURITY": "SEC",
    "ACCOUNT_ACCESS_REGISTRATION": "ACC",
    "FEATURE_FUNCTIONALITY": "FEA",
    "UI_UX": "UIX",
    "CUSTOMER_SERVICE": "CSV",
    "FEES_CHARGES": "FEE",
    "TRANSACTION_PERFORMANCE": "TRX",
    "APP_TECHNICAL_PERFORMANCE": "APP",
}
CODE_ASPECT = {v: k for k, v in ASPECT_CODE.items()}

POLARITY_CODE = {"POSITIVE": "P", "NEGATIVE": "N", "NEUTRAL": "U"}
CODE_POLARITY = {v: k for k, v in POLARITY_CODE.items()}

EXCLUSION_CODE = {
    "OUT_OF_SCOPE": "OS",
    "NO_SPECIFIC_ASPECT": "NS",
    "OUT_OF_ONTOLOGY": "OO",
    "INSUFFICIENT_CONTEXT": "IC",
    "UNINTERPRETABLE": "UN",
    "SPAM_FAKE": "SP",
}
CODE_EXCLUSION = {v: k for k, v in EXCLUSION_CODE.items()}


# ---------------------------------------------------------------- ABSTAIN (§5)
# "Jika aspect yang sama memiliki positive dan negative tanpa polarity yang jelas
# dominan, gunakan ABSTAIN dengan reason POLARITY_CONFLICT_CASE; conflict bukan
# kelas utama." -> ABSTAIN adalah nilai polaritas, bukan kelas sentimen.
ABSTAIN = "ABSTAIN"
ABSTAIN_REASON = "POLARITY_CONFLICT_CASE"
POLARITY_CODE[ABSTAIN] = "X"
CODE_POLARITY["X"] = ABSTAIN

# SPAM (§7 EC7): TANPA detector tervalidasi, spam TIDAK BOLEH jadi exclusion.
# Kode SP dari model diperlakukan sebagai FLAG, bukan alasan pembuangan.
SPAM_FLAG = "SPAM_CANDIDATE"
EXCLUSION_CODE.pop("SPAM_FAKE", None)
CODE_EXCLUSION.pop("SP", None)
EXCLUSIONS = [e for e in EXCLUSIONS if e != "SPAM_FAKE"]

# Nama tampilan untuk output akhir (§9 memakai "Transaction Performance")
ASPECT_DISPLAY = {
    "SECURITY": "Security",
    "ACCOUNT_ACCESS_REGISTRATION": "Account Access & Registration",
    "FEATURE_FUNCTIONALITY": "Feature & Functionality",
    "UI_UX": "UI / UX",
    "CUSTOMER_SERVICE": "Customer Service",
    "FEES_CHARGES": "Fees & Charges",
    "TRANSACTION_PERFORMANCE": "Transaction Performance",
    "APP_TECHNICAL_PERFORMANCE": "App / Technical Performance",
}
POLARITY_DISPLAY = {"POSITIVE": "Positive", "NEGATIVE": "Negative",
                    "NEUTRAL": "Neutral", ABSTAIN: "Abstain"}

PROMPT_VERSION = "absa_v1"


# ------------------------------------------------- definisi versi Inggris (v2)
# Prompt memakai instruksi Inggris atas data Indonesia; lihat docstring _rules()
# di prompt.py untuk dasar rujukannya (Hellwig et al., ESWA 2024 / LREC 2026).
ASPECT_DEF_EN = {
    "SECURITY":
        "Safety of funds and personal data: fraud, scams, hacked or compromised "
        "accounts, unauthorised transactions, data leaks, financial loss from cybercrime.",
    "ACCOUNT_ACCESS_REGISTRATION":
        "Account access: login, logout, registration, OTP, PIN, identity verification, "
        "username, password, email, phone number, blocked or frozen accounts.",
    "FEATURE_FUNCTIONALITY":
        "Availability, completeness and functioning of features, menus and services: "
        "PayLater, loans, QRIS as a feature, vouchers, promos as a feature, features "
        "removed after an update.",
    "UI_UX":
        "Appearance and ease of use: visual design, layout, navigation, scrolling, "
        "buttons, icons, ease or difficulty of finding things, ads that disrupt the interface.",
    "CUSTOMER_SERVICE":
        "Customer support: CS responsiveness, call centre, help centre, complaints, "
        "follow-up on issues, support chatbots.",
    "FEES_CHARGES":
        "Costs and deductions: admin fees, transfer fees, service charges, balance "
        "deducted as a fee, interest, penalties, pricing.",
    "TRANSACTION_PERFORMANCE":
        "Execution of transactions: transfers, top-ups, payments, bills, QRIS as a "
        "transaction, sending money, pending/failed/successful status, funds not "
        "arriving, balance deducted while the transaction failed, transaction speed.",
    "APP_TECHNICAL_PERFORMANCE":
        "Technical performance of the app: errors, crashes, force close, slowness, lag, "
        "long loading, maintenance, server outages, connectivity, memory usage, bugs, "
        "failed updates.",
}

BOUNDARIES_EN = [
    "A transfer menu that is hard to FIND -> UI_UX, not FEATURE_FUNCTIONALITY.",
    "A FAILED transaction with no technical cause mentioned -> TRANSACTION_PERFORMANCE "
    "only; do not add APP_TECHNICAL_PERFORMANCE.",
    "App fails to open because of the server -> APP_TECHNICAL_PERFORMANCE, not SECURITY.",
    "Cannot log in -> ACCOUNT_ACCESS_REGISTRATION. Add SECURITY only when there is a "
    "sign of fraud, account takeover or hacking.",
    "Balance deducted as a FEE -> FEES_CHARGES. Balance deducted while the transaction "
    "FAILED -> TRANSACTION_PERFORMANCE.",
    "A failed transaction with no mention of support -> do not assign CUSTOMER_SERVICE.",
    "A feature that is unavailable or missing -> FEATURE_FUNCTIONALITY. A feature that "
    "exists but errors technically -> APP_TECHNICAL_PERFORMANCE.",
]
