import json
from llm.groq_runtime import GroqRunTime
from llm.gpt_runtime import GPTRunTime

def explanation(claim, classification, news_scrape, title, evidence_link, content):
    groq_runtime = GroqRunTime()
    gpt_runtime = GPTRunTime()

    system_prompt = """
Kamu adalah asisten AI untuk verifikasi claim berbasis bukti dari internet.

Tugasmu:
1. Pahami inti claim yang diberikan.
2. Bandingkan claim tersebut dengan bukti dari internet yang tersedia, termasuk judul referensi, isi hasil scraping, dan link sumber.
3. Gunakan hasil klasifikasi model hanya sebagai referensi tambahan, bukan dasar utama.
4. Tentukan secara internal apakah claim tergolong "hoaks" atau "valid" berdasarkan kesesuaian fakta dengan bukti yang tersedia.

Aturan penilaian:
- Claim tergolong "valid" jika inti informasinya didukung oleh bukti internet yang relevan dan konsisten.
- Claim tergolong "hoaks" jika inti informasinya dibantah oleh bukti internet, tidak sesuai dengan fakta pada bukti, menyesatkan, atau keluar dari konteks.
- Jika bukti lemah, ambigu, tidak lengkap, atau hanya sebagian relevan, gunakan kesimpulan yang paling hati-hati berdasarkan bukti yang ada.
- Jangan hanya mengandalkan kemiripan kata; fokus pada kecocokan makna, fakta, konteks, aktor, waktu, lokasi, dan kronologi jika tersedia.
- Jika hasil klasifikasi model bertentangan dengan bukti internet, prioritaskan bukti internet.

Aturan output:
- Output harus berupa satu string penjelasan yang rapi dan mudah dibaca.
- Jangan output JSON.
- Gunakan format markdown ringan agar readable.
- Gunakan line break dengan pemisahan antarbagian.
- Gunakan format berikut:

**Hasil Verifikasi: [Hoaks/Valid]**

**Claim**
[tuliskan inti claim]

**Analisis Bukti**
[jelaskan apa yang disampaikan bukti internet dan apakah mendukung atau membantah claim]

**Kesimpulan**
[simpulkan secara eksplisit mengapa claim ini hoaks atau valid]

- Wajib sebutkan secara eksplisit apakah claim ini "hoaks" atau "valid".
- Penjelasan harus rinci, spesifik, dan berbasis bukti.
- Jangan menambahkan fakta di luar input yang diberikan.
"""
    user_prompt = f"""
Berikut adalah claim yang ingin diverifikasi:

Claim:
{claim}

Bukti dari internet yang ditemukan:

Judul referensi:
{title}

Link bukti:
{json.dumps(evidence_link, ensure_ascii=False, indent=2)}

Hasil salah satu scraping referensi:
{json.dumps(news_scrape, ensure_ascii=False, indent=2)}

Hasil klasifikasi model kami:
{json.dumps(classification, ensure_ascii=False, indent=2)}

Catatan:
- Hasil klasifikasi model bisa salah, jadi jangan dijadikan dasar utama.
- Prioritaskan bukti internet saat menentukan hasil akhir.

Instruksi:
- Fokus utama adalah memverifikasi claim, bukan hanya menilai kemiripan topik.
- Bandingkan inti claim dengan isi bukti internet.
- Tentukan apakah bukti mendukung atau membantah claim.
- Jika ada ketidaksesuaian, jelaskan bagian fakta, konteks, aktor, waktu, lokasi, atau kronologi yang tidak sesuai.
- Jika claim didukung oleh bukti, jelaskan bagian mana yang selaras.
- Sebutkan secara eksplisit apakah claim ini "hoaks" atau "valid".
- Jawaban harus berupa satu string yang rapi, readable, dan mengikuti format section yang diminta.

Berikan hanya penjelasan akhir.
"""

    # response = groq_runtime.generate_response(system_prompt, user_prompt)
    response = gpt_runtime.generate_response(system_prompt, user_prompt)
    return response