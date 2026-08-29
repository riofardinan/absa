# Anotasi ABSA IFRAD — 3 LLM lokal, korpus penuh

**Skema:** 8 kategori aspek × {POSITIVE, NEGATIVE, NEUTRAL}, satu pass per model,
greedy (temperature 0). EC6/NON_OPINION nonaktif karena skema memakai Neutral.

**Jalur:** in-process via `LLM.generate()` — tanpa HTTP server, tanpa subprocess,
tanpa ZMQ. Menghindari `NVML_SUCCESS assert` yang menggagalkan `vllm serve` di
driver 525, dan untuk job batch lebih cepat karena scheduler vLLM melihat seluruh
gelombang prompt sekaligus.

---

## LANGKAH 0 — work list  *(sudah dijalankan, ~40 detik)*

```bash
python3 prepare.py ../fintech_reviews_curated.csv
```
462.796 baris → 1.546 dibuang aturan deterministik (kosong/emoji-only/malformed)
→ **415.480 unit unik** (dedup normalized, hemat 9,9%).

Menghasilkan `work.jsonl` (yang dianotasi), `members.json` (propagasi label balik
ke tiap reviewId), `ruled.jsonl` (hasil aturan, sudah final).

---

## LANGKAH 1 — cek panjang token  *(30 detik, per model, JANGAN dilewati)*

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

python3 annotate_offline.py --tag qwen    --model Qwen/Qwen2.5-7B-Instruct        --check-tokens
python3 annotate_offline.py --tag mistral --model mistralai/Mistral-7B-Instruct-v0.2 --check-tokens
python3 annotate_offline.py --tag yi      --model 01-ai/Yi-1.5-9B-Chat-16K        --check-tokens
```
Hanya memuat tokenizer, tidak menyentuh GPU. Harus keluar `OK, sisa headroom ...`.
Kalau `!! MELAMPAUI`: naikkan `--max-model-len` atau turunkan `--chunk`.

Terukur pada Qwen2.5-7B @ chunk-25: input median 2.840 / p99 3.363 / **max 3.460**
tok, + ~1.000 tok output = **4.460**. Karena itu `--max-model-len` default **8192**,
bukan 4096.

---

## LANGKAH 2 — pilot 2.000 unit  *(~15 menit, per model)*

```bash
python3 annotate_offline.py --tag qwen    --model Qwen/Qwen2.5-7B-Instruct        --limit 2000
python3 annotate_offline.py --tag mistral --model mistralai/Mistral-7B-Instruct-v0.2 --limit 2000
python3 annotate_offline.py --tag yi      --model 01-ai/Yi-1.5-9B-Chat-16K        --limit 2000
```

Dua angka yang menentukan, dibaca dari baris SELESAI:

| Angka | Ambang | Kalau meleset |
|---|---|---|
| `fail %` | **< 0,5%** | turunkan `--chunk` ke 15 |
| `unit/dtk` | 415.480 ÷ angka ini = detik per model | kalau total > 5–6 jam, naikkan `--chunk` ke 40–50 lalu ULANGI LANGKAH 1 |

Lalu lihat agreement pilotnya:
```bash
python3 agreement.py ann_qwen.jsonl ann_mistral.jsonl ann_yi.jsonl
```
Kalau satu model kappanya jauh di bawah dua lainnya, **tukar model itu sekarang** —
jangan setelah 5 jam.

---

## LANGKAH 3 — produksi

```bash
rm -f ann_qwen.jsonl ann_mistral.jsonl ann_yi.jsonl    # buang hasil pilot dulu
CHUNK=25 MML=8192 bash run_all.sh
```
Berurutan: qwen → mistral → yi → skor agreement. Tiap model dapat penuh 40 GB.

**Aman diputus.** Jalankan ulang perintah yang sama; chunk yang sudah tertulis
dilewati (checkpoint dibaca dari `ann_*.jsonl` itu sendiri, di-`fsync` tiap gelombang).

---

## LANGKAH 4 — agreement + antrian adjudikasi

```bash
python3 agreement.py ann_qwen.jsonl ann_mistral.jsonl ann_yi.jsonl | tee agreement_report.txt
```
Skor per-SLOT (415.480 × 8 = **3.323.840 slot**, tiap slot 4-arah
{ABSENT, POSITIVE, NEGATIVE, NEUTRAL}), mengikuti Wittlinger et al. yang menskor
per-variabel, bukan per-dokumen. Exact-match tingkat review dengan 8 aspek akan
menandai hampir semua review "disagree" dan angkanya tidak berguna.

Menghasilkan:
- `adjudicate_tier3_manusia.jsonl` — ada slot 1/1/1 (ketiga model beda). **Prioritas.**
- `adjudicate_tier2_vote.jsonl` — ada slot 2/1, diselesaikan majority vote.
- Tier 1 (semua slot bulat) diterima tanpa review.

---

## Model

| tag | model | keluarga | ctx | preseden |
|---|---|---|---|---|
| qwen | `Qwen/Qwen2.5-7B-Instruct` | Alibaba | 32k | MoLLIA (varian instruct, **bukan Coder**) |
| mistral | `mistralai/Mistral-7B-Instruct-v0.2` | Mistral | 32k | MoLLIA — **model persis** |
| yi | `01-ai/Yi-1.5-9B-Chat-16K` | 01.AI | 16k | MoLLIA — **model persis** |

Ketiganya terbuka (tidak gated), tanpa sliding window, BF16 muat di 40 GB.

⚠️ **Jangan pakai `01-ai/Yi-1.5-9B-Chat` biasa** — konteksnya hanya **4.096**,
tidak muat chunk-25 (butuh 4.460). Harus varian `-16K`.

---

## Kalau bermasalah

| Gejala | Tindakan |
|---|---|
| `NVML_SUCCESS assert` | `bash fix_nvml.sh` — langkah 4 vs 5 memisahkan penyebabnya |
| OOM saat muat model | turunkan `--gpu-util` ke 0.85 |
| crash aneh di driver lama | tambahkan `--enforce-eager` (matikan CUDA graph) |
| `fail %` tinggi | turunkan `--chunk`; cek juga `--max-tokens` cukup (≥ chunk × 40) |
| model punya sliding window | tambahkan `--no-prefix-caching` |
| rugi besar saat crash | turunkan `--wave` ke 300 |

`check_server.sh` — cek driver/torch/vLLM. `fix_nvml.sh` — diagnosis NVML.

---

## Yang WAJIB masuk paper

- **Multi-item prompting (25 review/call) adalah penyimpangan.** Hellwig et al.
  (LREC 2026), Wittlinger et al. (medRxiv 2026), dan Yuan et al. (MCHR) semuanya
  1-item-per-call. Validasi: jalankan `--chunk 1 --limit 500 --tag qwen_single`,
  bandingkan dengan 500 unit pertama hasil chunk-25, laporkan agreement-nya.
- **Konsensus 3/3 bukan bukti kebenaran.** Wittlinger menemukan inter-LLM κ
  (0,83–0,89) > LLM-to-human κ (0,82–0,84): error antar-model berkorelasi.
  Karena itu adjudikasi manusia tetap perlu meski agreement tinggi.
- **Laporkan raw agreement DAN Fleiss κ**, keseluruhan dan per-aspek. 58,61%
  korpus tidak men-trigger satu pun seed keyword → mayoritas slot unanimous ABSENT
  dan menggelembungkan raw agreement (paradoks kappa; Artstein & Poesio).
- **UI/UX hanya 0,82% korpus** (3.790 review) — CI per-aspeknya lebar. Sebutkan.
- **Neutral titik terlemah LLM** (Martin et al. 2026: κ sentimen 0,65; F1 positif
  0,84 / negatif 0,78, jeblok di neutral/ambiguous). Disagreement menumpuk di sana.
- **Volume adjudikasi.** Wittlinger tidak mengadjudikasi semua disagreement:
  sampel stratified 377 kasus yang meng-oversample stratum agreement rendah, lalu
  ekstrapolasi prevalence-adjusted. Tiru itu — mengadjudikasi seluruh Tier 3 dari
  415.480 unit tidak akan selesai.
- **Sebutkan snapshot model, tanggal eksperimen, temperature 0, chunk size,
  dan versi vLLM** agar reproducible.
