import json
from llm.groq_runtime import GroqRunTime
from llm.gpt_runtime import GPTRunTime

def claim_check(claim, evidence_link):
    groq_runtime = GroqRunTime()
    gpt_runtime = GPTRunTime()

    system_prompt = """
Kamu adalah asisten AI yang memverifikasi apakah daftar link berita relevan terhadap sebuah klaim.

Aturan penilaian:
1. Periksa SEMUA link satu per satu.
2. Gunakan penilaian SEMANTIK, bukan sekadar kecocokan kata.  
   Artinya:
   - Jika isi berita membahas isu yang sama, makna yang setara, topik identik, atau rumor yang sama, anggap relevan.
   - Kata-kata berbeda tidak masalah selama maknanya selaras dengan klaim.
3. Jika terdapat MINIMAL SATU link yang relevan secara makna dengan klaim, keluarkan **TEPAT SATU KATA**:
   sesuai
   (tanpa teks tambahan apa pun).
4. Jika TIDAK ADA satu pun link yang relevan secara makna, keluarkan:
   tidak sesuai: <alasan singkat 1–3 kalimat>
5. Jangan membuat daftar link, jangan menjelaskan tiap link, jangan menambah format apa pun.

Definisi “relevan secara makna”:
- Isi link membahas hal yang secara langsung terkait dengan inti klaim, meskipun menggunakan kata, sudut pandang, atau istilah berbeda.
- Contoh: klaim “prabowo berbiji satu” relevan dengan berita yang membahas isu Prabowo dikebiri, rumor alat kelamin, atau pembahasan dugaan kondisi organ reproduksi Prabowo — karena maknanya sama atau sangat dekat.

Ikuti format output secara ketat.

"""

    user_prompt = f"""
Berikut klaim yang ingin diperiksa:
{claim}

Berikut daftar link berita (10 link):
{json.dumps(evidence_link, ensure_ascii=False, indent=2)}

Tentukan apakah minimal satu link relevan secara makna dengan klaim, lalu berikan output sesuai aturan.

"""

    # response = groq_runtime.generate_response(system_prompt, user_prompt)
    response = gpt_runtime.generate_response(system_prompt, user_prompt)

    # pastikan output bersih (opsional sanity-check)
    cleaned = response.strip()
    return cleaned
