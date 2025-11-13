import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import json
import re
from googleapiclient.discovery import build
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from llm.groq_runtime import GroqRunTime

# ======================
# KONFIGURASI
# ======================

MODEL_DIR = "./saved_models/IndoBERT_version1/32_2e-05"

BAD_EXT = [".pdf", ".zip", ".doc", ".docx"]
ANTI_BOT_KEYWORDS = ["just a moment", "captcha", "cloudflare"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)


# ======================
# GOOGLE CUSTOM SEARCH
# ======================
def google_search(query, total_results=20, api_key=API_KEY, cse_id=CSE_ID):
    service = build("customsearch", "v1", developerKey=api_key)
    results = []
    start = 1

    while len(results) < total_results:
        num = min(10, total_results - len(results))
        try:
            res = service.cse().list(q=query, cx=cse_id, num=num, start=start).execute()
        except Exception as e:
            print(f"⚠️ Google CSE error: {e}")
            break

        items = res.get('items', [])
        if not items:
            break

        for item in items:
            url = item['link']
            if any(url.lower().split("?")[0].endswith(ext) for ext in BAD_EXT):
                continue
            results.append(url)

        start += num
    return results


# ======================
# SCRAPE HTML
# ======================
def scrape_html(url):
    # Gunakan user-agent custom agar tidak diblokir situs berita
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; HoaxDetector/1.0; +https://example.com)"
    }

    try:
        # Tambahkan headers dan timeout
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            raise Exception(f"Status code {response.status_code}")

        html_lower = response.text.lower()
        if any(key in html_lower for key in ANTI_BOT_KEYWORDS):
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        judul = soup.title.string.strip() if soup.title and soup.title.string else "Unknown"
        time_tag = soup.find("time")
        tanggal = time_tag.get_text(strip=True) if time_tag else "Unknown"

        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]

        # Cari div kandidat berisi artikel
        div_candidates = soup.find_all("div", class_=re.compile("(article|content|post|isi|entry)"))
        for div in div_candidates:
            paragraphs.extend([p.get_text(" ", strip=True) for p in div.find_all("p")])

        # Filter paragraf pendek
        paragraphs = [p for p in paragraphs if len(p) > 50]
        content = "\n".join(paragraphs) if paragraphs else "Tidak berhasil ekstrak isi artikel"

        return {
            "judul": judul,
            "tanggal": tanggal,
            "sumber": urlparse(url).netloc.replace("www.", ""),
            "link": url,
            "content": content
        }

    except Exception as e:
        print(f"⚠️ Gagal scrape {url}: {e}")
        return None



# ======================
# KLASIFIKASI INDO BERT
# ======================
def classify_berita(title, content):
    text = f"{title}\n\n{content}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

    label = "hoaks" if pred == 1 else "valid"
    confidence = round(probs[0][pred].item() * 100, 2)
    return {"label": label, "confidence": confidence, "probs": probs.tolist()}


# ======================
# PENILAIAN LLM (Groq)
# ======================
def llm_judge(classification, news_scrape, title, content):
    groq_runtime = GroqRunTime()

    system_prompt = """Kamu adalah asisten AI yang menilai apakah suatu berita tergolong hoaks atau valid.
Gunakan hasil klasifikasi IndoBERT dan hasil scraping berita dari internet sebagai referensi.
Berikan output JSON dengan atribut:
- "label": "hoaks" atau "valid"
- "confidence": (0-100)
- "reason": penjelasan singkat."""

    user_prompt = f"""
Berikut adalah berita yang ingin diklasifikasikan:

Judul: {title}
Isi: {content}

Hasil klasifikasi IndoBERT:
{json.dumps(classification, ensure_ascii=False, indent=2)}

Hasil scraping berita referensi:
{json.dumps(news_scrape, ensure_ascii=False, indent=2)}

Tentukan apakah berita ini hoaks atau valid berdasarkan konteks dan kesesuaian dengan berita referensi.
"""
    response = groq_runtime.generate_response(system_prompt, user_prompt)
    return response


# ======================
# PIPELINE UTAMA
# ======================
def pipeline(title, content):
    print(f"🔍 Mencari berita terkait dengan judul: '{title}' ...")
    links = google_search(title, total_results=20)

    results = []
    seen_domains = set()

    for url in links:
        domain = urlparse(url).netloc.replace("www.", "")
        if domain in seen_domains:
            continue
        print(f"📰 Scraping {url} ...")
        scraped = scrape_html(url)
        if scraped:
            results.append(scraped)
            seen_domains.add(domain)
        if len(results) >= 3:
            break

    if not results:
        print("❌ Tidak ada berita valid yang bisa di-scrape.")
        return None

    # IndoBERT klasifikasi input user (title + content)
    classification = classify_berita(title, content)

    # LLM penilaian akhir
    llm_result = llm_judge(classification, results, title, content)

    final_output = {
        "input_user": {"title": title, "content": content},
        "classification": classification,
        "scraped_articles": results,
        "llm_judgement": llm_result
    }

    print(json.dumps(final_output, indent=2, ensure_ascii=False))
    return final_output


# ======================
# MAIN EKSEKUSI
# ======================
if __name__ == "__main__":
    # contoh: user input manual
#     title = "Purbaya Sidak Bea Cukai, Kaget Barang Rp117 Ribu Dijual Rp50 Juta"
#     content = """Jakarta, CNN Indonesia -- Menteri Keuangan Purbaya Yudhi Sadewa melakukan inspeksi mendadak (sidak) ke Kantor Pengawasan dan Pelayanan Bea Cukai Tipe Madya Pabean (KPPBC TMP) Tanjung Perak. Dalam sidak itu, ia mengecek sejumlah barang impor. Purbaya kaget saat melihat barang yang cukup canggih hanya dilaporkan dengan harga US$7 atau sekitar Rp117 ribu. Harganya Rp100 ribu, gila murah banget. Ini Rp50 jutaan di pasar, berarti mereka ambil untung gede ya, kata Purbaya dalam sidak yang diunggah di akun Tiktok @purbayayudhis, Rabu (13/11). Purbaya menilai barang itu terlalu bagus untuk harga yang diklaim di dokumen pengiriman. Oleh karenanya, ia meminta Kantor Balai Laboratorium Bea dan Cukai (KBLBC) Kelas II Surabaya mengecek ulang. Saya bilang ke teman-teman lab kalau kurang peralatan kasih tahu sehingga bisa kita lengkapin, saya juga lihat container scanner, baru dua minggu sudah banyak dipasang, ucapnya. Purbaya berniat meningkatkan fasilitas teknologi untuk Bea Cukai. Ia mau semua pemeriksaan bisa dipantau langsung oleh Kementerian Keuangan. Ini kan IT-based, saya akan tarik ke Jakarta sehingga orang Jakarta bisa lihat langsung apa yang terjadi di lapangan, ucapnya.

# """
    title = "Aturan Tilang 2026: Diberlakukan Denda Manual 150%"
    content = """Pada Selasa (28/10/2025), beredar sebuah video (arsip cadangan) di Facebook oleh akun “MDG99” (https://www.facebook.com/profile.php?id=61571511892184) dengan narasi:  Selamat ya pak , semoga berkah 🤲 di post, dan narasi:  Seorang dosen mengundurkan diri setelah cair di situs MDG99  Saya mengajukan untuk mengundurkan diri dari pekerjaan sebagai dosen mulai hari ini karena saya baru saja mendapatkan jp paus yang cukup luar biasa dari situs MDG99 Gini gini pak, awalnya saya itu cuma sebagai iseng saja waktu lihat di iklan kayaknya seru gitu, ya udah dicoba dan saya sama istri pun kaget karena depo kecil aja malah bisa dapat puluhan juta Dan besoknya saya nulis surat kepada atasan saya bahwa saya mundur, ya saya udah tua juga sudah capek kerja untungnya juga ada situs MDG99 yang bisa bantu saya untuk kasih makan keluarga juga ya kan Saya tidak ada jam ngajar saya hari ini, sampai saya menulisnya selamanya Saya mundur mulai hari ini sampai dengan selamanya di dalam video.  Per tangkapan layar dibuat unggahan tersebut  sudah mendapatkan 68 komentar, dibagikan ulang 3 kali, dan disukai oleh 174 pengguna Facebook lainnya. 

"""
    pipeline(title, content)
