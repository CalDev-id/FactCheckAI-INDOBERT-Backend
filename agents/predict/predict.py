import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from dotenv import load_dotenv
from llm.gpt_runtime import GPTRunTime
import re
import json
load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR")
if not MODEL_DIR:
    raise ValueError("❌ MODEL_DIR tidak ditemukan di file .env")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

def classify_berita(title, content):
    text = f"{title}\n\n{content}"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,   
        padding="max_length", 
        max_length=512      
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

    label = "valid" if pred == 1 else "hoaks"
    confidence = round(probs[0][pred].item() * 100, 2)

    # return {"label": label, "confidence": confidence, "probs": probs.tolist()}
    return {"label": label, "confidence": confidence}

def advance_classify_berita(claim, classification, titles, news_list):
    """
    news_list = [
        {"title": "...", "content": "..."},
        {"title": "...", "content": "..."},
    ]
    """

    gpt_runtime = GPTRunTime()

    # 🔹 Extract title list dulu (buat prioritas awal LLM)
    titles = [news["title"] for news in news_list]

    system_prompt = """
Kamu adalah asisten AI untuk verifikasi klaim berita berbasis bukti dari internet.

Tugas utama:
1. Pahami inti claim secara menyeluruh.
2. Analisis daftar judul berita (minimal 10 judul) untuk memahami kecenderungan informasi secara global.
3. Setelah itu, validasi menggunakan isi konten berita hasil scraping.
4. Bandingkan claim dengan bukti:
   - Apakah didukung?
   - Dibantah?
   - Tidak relevan / ambigu?
5. Gunakan hasil klasifikasi model hanya sebagai referensi tambahan.

========================================
URUTAN ANALISIS WAJIB (JANGAN DILANGGAR)
========================================
1. Analisis judul berita terlebih dahulu:
   - Identifikasi pola narasi (mayoritas mendukung / membantah / campuran)
   - Gunakan sebagai sinyal awal (BELUM final)

2. Analisis isi berita:
   - Validasi fakta dari judul menggunakan konten
   - Ambil poin penting dari beberapa berita
   - Cocokkan dengan claim

3. Ambil keputusan akhir berdasarkan isi berita (BUKAN judul saja)

========================================
ATURAN PENILAIAN
========================================
- "valid" → jika bukti mendukung claim
- "hoaks" → jika:
  - dibantah bukti, ATAU
  - tidak sesuai fakta / menyesatkan

Jika bukti:
- lemah
- ambigu
- tidak cukup
→ pilih kesimpulan paling konservatif

========================================
FORMAT OUTPUT (WAJIB JSON)
========================================
{
  "final_label": "hoaks|valid",
  "final_confidence": 0-100,
  "explanation": "markdown string"
}

========================================
FORMAT EXPLANATION (WAJIB)
========================================

**Hasil Verifikasi: [Hoaks/Valid]**

**Claim**
[ringkasan claim]

**Analisis Bukti**
[fakta dari beberapa konten berita + relevansi terhadap claim]

**Kesimpulan**
[kenapa hoaks atau valid]

Catatan:
- WAJIB eksplisit menyebut "hoaks" atau "valid"
- Jangan mengarang
- Jangan hanya bergantung pada judul
- Jangan output selain JSON
"""
    user_prompt = f"""
========================================
CLAIM
========================================
{claim}

========================================
DAFTAR 10 JUDUL BERITA (TAHAP 1 - OVERVIEW)
========================================
{json.dumps(titles, ensure_ascii=False, indent=2)}

========================================
DETAIL BERITA (TAHAP 2 - VALIDASI KONTEN)
========================================
{json.dumps(news_list, ensure_ascii=False, indent=2)}

========================================
HASIL MODEL (REFERENSI TAMBAHAN)
========================================
{json.dumps(classification, ensure_ascii=False, indent=2)}

========================================
INSTRUKSI
========================================

WAJIB IKUTI URUTAN INI:

1. Analisis 10 judul berita:
   - Apakah mayoritas mendukung atau membantah claim?
   - Apa pola narasi yang muncul?

2. Analisis isi berita:
   - Ambil fakta penting dari beberapa berita
   - Validasi apakah sesuai dengan claim
   - Cari perbedaan fakta, konteks, atau misleading

3. Bandingkan dengan claim:
   - Cocok / tidak cocok?

4. Gunakan hasil model hanya sebagai referensi tambahan

5. Tentukan:
   - "valid" atau "hoaks"

6. Buat explanation dalam format markdown sesuai aturan

========================================
OUTPUT
========================================

Kembalikan HANYA JSON valid:

{{
  "final_label":"hoaks|valid",
  "final_confidence":0-100,
  "explanation":"markdown explanation"
}}
"""

    raw = gpt_runtime.generate_response(system_prompt, user_prompt)

    if not raw:
        return {
            "final_label": "unknown",
            "final_confidence": 0,
            "explanation": "Tidak ada respons dari AI"
        }

    text = raw.strip()

    # 1) Direct parse
    try:
        return json.loads(text)
    except:
        pass

    # 2) Extract JSON
    try:
        json_str = re.search(r"\{.*\}", text, re.DOTALL).group(0)
        return json.loads(json_str)
    except:
        pass

    # 3) fallback
    print(system_prompt)
    print(user_prompt)
    return {
        "final_label": "unknown",
        "final_confidence": 0,
        "explanation": f"Failed parsing: {text}"
    }


#     gpt_runtime = GPTRunTime()

#     system_prompt = """
# Kamu adalah asisten AI untuk verifikasi klaim berita berbasis bukti.

# Tugas utama:
# 1. Pahami inti claim yang diberikan.
# 2. Bandingkan claim dengan bukti yang ditemukan dari internet.
# 3. Nilai apakah bukti tersebut mendukung, membantah, atau tidak cukup relevan terhadap claim.
# 4. Gunakan hasil klasifikasi IndoBERT hanya sebagai sinyal tambahan, bukan sumber kebenaran utama.
# 5. Fokus pada kesesuaian fakta antara claim dan bukti, bukan hanya kemiripan kata.

# Aturan penilaian:
# - Label "valid" jika bukti internet secara jelas mendukung inti claim.
# - Label "hoaks" jika bukti internet secara jelas membantah claim, atau claim mengandung informasi yang tidak sesuai dengan bukti.
# - Jika bukti lemah, ambigu, tidak relevan, terlalu sedikit, atau saling bertentangan, pilih label yang paling konservatif berdasarkan bukti yang ada.
# - Jangan terlalu bergantung pada satu judul; pertimbangkan isi konten bukti.
# - Jika hasil IndoBERT bertentangan dengan bukti, prioritaskan bukti.

# Output wajib:
# - Kembalikan HANYA JSON valid.
# - Jangan menambahkan markdown, komentar, atau teks lain.
# - Gunakan format persis:
# {{"final_label":"hoaks|valid","final_confidence":0-100}}

# Aturan confidence:
# - 85-100: bukti sangat kuat dan konsisten
# - 70-84: bukti cukup kuat
# - 50-69: bukti sedang / ada sedikit ambiguitas
# - 0-49: hanya jika tetap terpaksa memilih label meski bukti lemah

# Jangan mengarang fakta di luar input yang diberikan.
# """
#     user_prompt = f"""
# Verifikasi claim berikut berdasarkan bukti internet yang tersedia.

# Claim:
# {claim}

# Bukti internet yang ditemukan:
# Judul: {title}
# Konten: {content}

# Link sumber bukti:
# {json.dumps(evidence_link, ensure_ascii=False)}

# Hasil klasifikasi IndoBERT (hanya referensi tambahan, bisa salah):
# {json.dumps(classification, ensure_ascii=False, indent=2)}

# Instruksi:
# - Bandingkan inti claim dengan isi bukti.
# - Tentukan apakah bukti mendukung atau membantah claim.
# - Utamakan isi bukti dibanding hasil IndoBERT.
# - Berikan keputusan akhir dalam JSON saja.

# Output:
# {{"final_label":"hoaks|valid","final_confidence":0-100}}
# """

#     # response = groq_runtime.generate_response(system_prompt, user_prompt)
#     raw = gpt_runtime.generate_response(system_prompt, user_prompt)

#     if not raw:
#         return {
#             "final_label": "unknown",
#             "final_confidence": 0,
#             "error": "Tidak ada respons dari AI"
#         }

#     text = raw.strip()

#     # 1) Coba parsing langsung
#     try:
#         parsed = json.loads(text)
#         return parsed   # ✔ dict
#     except json.JSONDecodeError:
#         pass

#     # 2) Jika output berisi text + JSON → extract JSON-nya
#     try:
#         json_str = re.search(r"\{.*\}", text, re.DOTALL).group(0)
#         parsed = json.loads(json_str)
#         return parsed
#     except:
#         pass

#     # 3) Fallback → kalau outputnya salah, tetap return JSON valid
#     return {
#         "final_label": "unknown",
#         "final_confidence": 0,
#         "error": f"Failed to parse GPT response: {text}"
#     }