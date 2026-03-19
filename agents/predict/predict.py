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

def advance_classify_berita(claim, classification, news_scrape, title, evidence_link, content):
    gpt_runtime = GPTRunTime()

#     system_prompt = """Kamu adalah asisten AI yang menilai apakah suatu berita tergolong hoaks atau valid.
# Gunakan hasil klasifikasi IndoBERT dan hasil scraping berita dari internet sebagai referensi.
# tentukan apakah berita ini hoaks atau valid berdasarkan konteks dan kesesuaian dengan berita referensi.
# output dalam format json dengan key: final_label (hoaks/valid) dan final_confidence (0-100)
# hanya berikan output json saja, tanpa penjelasan tambahan.
# contoh output:
# {"final_label": "hoaks", "final_confidence": 85.5}
# """

#     user_prompt = f"""
# Berikut adalah berita yang ingin diklasifikasikan:

# Claim: {claim}

# bukti yang ditemukan di internet terkait claim ini:
# Judul: {title}
# content: {content}

# Hasil klasifikasi IndoBERT:
# {json.dumps(classification, ensure_ascii=False, indent=2)}
# (kadang hasil klasifikasi bisa salah, jadi tentukan berdasarkan bukti yang ada)

# Link bukti yang ditemukan:
# {json.dumps(evidence_link, ensure_ascii=False, indent=2)}

# Tentukan apakah berita ini hoaks atau valid berdasarkan konteks dan kesesuaian dengan berita referensi.
# """

    system_prompt = """
Kamu adalah asisten AI untuk verifikasi klaim berita berbasis bukti.

Tugas utama:
1. Pahami inti claim yang diberikan.
2. Bandingkan claim dengan bukti yang ditemukan dari internet.
3. Nilai apakah bukti tersebut mendukung, membantah, atau tidak cukup relevan terhadap claim.
4. Gunakan hasil klasifikasi IndoBERT hanya sebagai sinyal tambahan, bukan sumber kebenaran utama.
5. Fokus pada kesesuaian fakta antara claim dan bukti, bukan hanya kemiripan kata.

Aturan penilaian:
- Label "valid" jika bukti internet secara jelas mendukung inti claim.
- Label "hoaks" jika bukti internet secara jelas membantah claim, atau claim mengandung informasi yang tidak sesuai dengan bukti.
- Jika bukti lemah, ambigu, tidak relevan, terlalu sedikit, atau saling bertentangan, pilih label yang paling konservatif berdasarkan bukti yang ada.
- Jangan terlalu bergantung pada satu judul; pertimbangkan isi konten bukti.
- Jika hasil IndoBERT bertentangan dengan bukti, prioritaskan bukti.

Output wajib:
- Kembalikan HANYA JSON valid.
- Jangan menambahkan markdown, komentar, atau teks lain.
- Gunakan format persis:
{{"final_label":"hoaks|valid","final_confidence":0-100}}

Aturan confidence:
- 85-100: bukti sangat kuat dan konsisten
- 70-84: bukti cukup kuat
- 50-69: bukti sedang / ada sedikit ambiguitas
- 0-49: hanya jika tetap terpaksa memilih label meski bukti lemah

Jangan mengarang fakta di luar input yang diberikan.
"""
    user_prompt = f"""
Verifikasi claim berikut berdasarkan bukti internet yang tersedia.

Claim:
{claim}

Bukti internet yang ditemukan:
Judul: {title}
Konten: {content}

Link sumber bukti:
{json.dumps(evidence_link, ensure_ascii=False)}

Hasil klasifikasi IndoBERT (hanya referensi tambahan, bisa salah):
{json.dumps(classification, ensure_ascii=False, indent=2)}

Instruksi:
- Bandingkan inti claim dengan isi bukti.
- Tentukan apakah bukti mendukung atau membantah claim.
- Utamakan isi bukti dibanding hasil IndoBERT.
- Berikan keputusan akhir dalam JSON saja.

Output:
{{"final_label":"hoaks|valid","final_confidence":0-100}}
"""

    # response = groq_runtime.generate_response(system_prompt, user_prompt)
    raw = gpt_runtime.generate_response(system_prompt, user_prompt)

    if not raw:
        return {
            "final_label": "unknown",
            "final_confidence": 0,
            "error": "Tidak ada respons dari AI"
        }

    text = raw.strip()

    # 1) Coba parsing langsung
    try:
        parsed = json.loads(text)
        return parsed   # ✔ dict
    except json.JSONDecodeError:
        pass

    # 2) Jika output berisi text + JSON → extract JSON-nya
    try:
        json_str = re.search(r"\{.*\}", text, re.DOTALL).group(0)
        parsed = json.loads(json_str)
        return parsed
    except:
        pass

    # 3) Fallback → kalau outputnya salah, tetap return JSON valid
    return {
        "final_label": "unknown",
        "final_confidence": 0,
        "error": f"Failed to parse GPT response: {text}"
    }