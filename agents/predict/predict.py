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

def advance_classify_berita(classification, news_scrape, title, evidence_link, content):
    gpt_runtime = GPTRunTime()

    system_prompt = """Kamu adalah asisten AI yang menilai apakah suatu berita tergolong hoaks atau valid.
Gunakan hasil klasifikasi IndoBERT dan hasil scraping berita dari internet sebagai referensi.
tentukan apakah berita ini hoaks atau valid berdasarkan konteks dan kesesuaian dengan berita referensi.
output dalam format json dengan key: final_label (hoaks/valid) dan final_confidence (0-100)
hanya berikan output json saja, tanpa penjelasan tambahan.
contoh output:
{"final_label": "hoaks", "final_confidence": 85.5}
"""

    user_prompt = f"""
Berikut adalah berita yang ingin diklasifikasikan:

Judul: {title}
content: {content}

Hasil klasifikasi IndoBERT:
{json.dumps(classification, ensure_ascii=False, indent=2)}
(kadang hasil klasifikasi bisa salah, jadi tentukan berdasarkan bukti yang ada)

Link bukti yang ditemukan:
{json.dumps(evidence_link, ensure_ascii=False, indent=2)}

Hasil scraping berita referensi:
{json.dumps(news_scrape, ensure_ascii=False, indent=2)}

Tentukan apakah berita ini hoaks atau valid berdasarkan konteks dan kesesuaian dengan berita referensi.
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