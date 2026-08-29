# Pipeline anotasi ABSA IFRAD — 3 LLM, korpus penuh

8 aspek × {POSITIVE, NEGATIVE, NEUTRAL}. Satu pass per model, greedy (temperature 0).
EC6/NON_OPINION nonaktif karena skema memakai Neutral (sesuai kondisi di dokumen).

## Urutan jalan

### 0. Work list (sudah dijalankan, ~40 detik)
    python3 prepare.py ../fintech_reviews_curated.csv
462.796 baris → 1.546 dibuang aturan → **415.480 unit unik** (hemat 9,9%).
Menghasilkan `work.jsonl`, `members.json` (propagasi label balik ke reviewId), `ruled.jsonl`.

### 1. PILOT dulu — 20 menit, jangan dilewati
    export API_KEY=...
    python3 annotate.py --tag pilotA --base-url <URL> --model <MODEL> \
        --chunk 25 --concurrency 24 --limit 2000
Yang harus kamu baca dari output: **chunk/menit** (untuk ETA sebenarnya),
**repair/split/fail** (kalau split tinggi, turunkan `--chunk`), dan **token in/out**
(untuk proyeksi biaya). Kalau `fail > 0,5%`, jangan lanjut ke produksi.

### 2. Produksi — 3 model paralel
    python3 annotate.py --tag A --base-url <URL_A> --model <M_A> --api-key-env KEY_A &
    python3 annotate.py --tag B --base-url <URL_B> --model <M_B> --api-key-env KEY_B &
    python3 annotate.py --tag C --base-url <URL_C> --model <M_C> --api-key-env KEY_C &
    wait
16.620 call/model @ chunk-25. Aman diputus: jalankan ulang perintah yang sama,
chunk yang sudah selesai dilewati (checkpoint dibaca dari file output sendiri).

### 3. Agreement + antrian adjudikasi
    python3 agreement.py ann_A.jsonl ann_B.jsonl ann_C.jsonl
Menghasilkan `adjudicate_tier3_manusia.jsonl` (prioritas) dan `adjudicate_tier2_vote.jsonl`.

## Penyetelan
| Gejala | Tindakan |
|---|---|
| kena 429 terus | turunkan `--concurrency` |
| `split` tinggi | turunkan `--chunk` ke 10–15 |
| output kepotong | naikkan `--max-tokens` |
| provider dukung JSON mode | tambahkan `--json-mode` |

## Yang WAJIB masuk paper
- **Multi-item prompting (25 review/call) adalah penyimpangan** dari Hellwig et al.
  (LREC 2026), Wittlinger et al. (medRxiv 2026), dan Yuan et al. (MCHR) — ketiganya
  1-item-per-call. Validasi dengan menjalankan `--chunk 1 --limit 500` lalu bandingkan
  labelnya terhadap hasil chunk-25 pada 500 unit yang sama; laporkan agreement-nya.
- Tiga model harus dari **keluarga pretraining berbeda**. Wittlinger menemukan
  inter-LLM κ (0,83–0,89) > LLM-to-human κ (0,82–0,84): error antar-model berkorelasi,
  jadi **konsensus 3/3 bukan bukti kebenaran**.
- Laporkan raw agreement **dan** Fleiss κ, keseluruhan dan per-aspek. 58,61% korpus
  tidak men-trigger satu pun seed keyword → mayoritas slot unanimous ABSENT dan
  menggelembungkan raw agreement (paradoks kappa; Artstein & Poesio).
- UI/UX hanya 0,82% korpus (3.790 review) — CI per-aspeknya akan lebar. Sebutkan.
- Neutral adalah titik terlemah LLM (Martin et al. 2026: κ sentimen 0,65; F1 positif
  0,84 / negatif 0,78, jeblok di neutral/ambiguous). Disagreement akan menumpuk di sana.
- Sebutkan snapshot model + tanggal eksperimen — sesuai catatanmu sendiri untuk M2.

---

# Jalur GRATIS

## Opsi 1 — A100 di server (rekomendasi: gratis DAN tanpa limit)

vLLM menyajikan endpoint OpenAI-compatible, jadi `annotate.py` dipakai APA ADANYA.
Tidak ada RPM, tidak ada RPD, tidak ada biaya.

Di server:
    pip install vllm
    vllm serve <model> --port 8000 --max-model-len 8192 \
        --gpu-memory-utilization 0.90 --enable-prefix-caching

Dari mana saja (atau di server itu sendiri):
    python3 probe.py --base-url http://localhost:8000/v1 --model <model> --chunk 25
    python3 annotate.py --tag A --base-url http://localhost:8000/v1 --model <model> \
        --chunk 25 --concurrency 64

`--enable-prefix-caching` penting: prefix prompt ~1.800 token identik di semua call,
dan hanya dihitung sekali.

Muat di A100 40GB (Ampere — **tanpa FP8 native**, pakai INT4/AWQ atau bf16):
| Ukuran | Presisi | ~VRAM |
|---|---|---|
| 27B | INT4 (AWQ/GPTQ) | 16–17 GB — setup Hellwig & Wittlinger |
| 32B | INT4 | 18–20 GB |
| 12–14B | bf16 | 24–28 GB |
| 7–9B | bf16 | 15–18 GB — **2 model muat bersamaan** |
| 70B | INT4 | 38–40 GB — TIDAK aman bersama KV cache |

Wittlinger et al.: di atas ~12B (Gemma 3) / ~14B (DeepSeek-R1), penambahan ukuran
model hanya memberi gain agreement marginal. Tidak perlu mengejar model terbesar.
Tiga model harus dari **keluarga pretraining berbeda**.

## Opsi 2 — free tier API

Yang menggigit adalah **RPD (request per hari)**, bukan RPM:

| chunk | call/model | RPM utk 5 jam | out tok/call |
|---:|---:|---:|---:|
| 25 | 16.620 | 55,4 | 1.000 |
| 50 | 8.310 | 27,7 | 2.000 |
| 100 | 4.155 | 13,8 | 4.000 |
| 200 | 2.078 | 6,9 | 8.000 — menabrak batas output banyak model |

RPM-nya ringan; 4.155 request/model/hari yang berat. **Ukur dulu, jangan percaya
dokumentasi** — `probe.py` melaporkan throughput nyata, header rate-limit, dan
proyeksi total dalam ~2 menit. Naikkan `--chunk` untuk menekan jumlah call, tapi
awasi kolom "output invalid": >5% berarti chunk kekecilan reliabilitasnya.

Karena butuh 3 model berbeda, sebar ke 3 provider berbeda — tiap provider hanya
melihat sepertiga beban, dan itu memang syarat keragaman keluarga model.
