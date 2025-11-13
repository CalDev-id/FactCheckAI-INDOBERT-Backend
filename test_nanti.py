# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# # Load model & tokenizer sekali saja saat server start
# MODEL_DIR = "./saved_models/IndoBERT_version1/32_2e-5"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# app = FastAPI()

# class InputText(BaseModel):
#     text: str

# @app.post("/predict")
# def predict(input_data: InputText):
#     inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#         pred = torch.argmax(probs, dim=-1).item()
#     return {
#         "label": pred,
#         "probabilities": probs.tolist()
#     }


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./saved_models/IndoBERT_version1/32_2e-05"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

text = "Aturan Tilang 2026: Diberlakukan Denda Manual 150% Pada Selasa (28/10/2025), beredar sebuah video (arsip cadangan) di Facebook oleh akun “MDG99” (https://www.facebook.com/profile.php?id=61571511892184) dengan narasi:  Selamat ya pak , semoga berkah 🤲 di post, dan narasi:  Seorang dosen mengundurkan diri setelah cair di situs MDG99  Saya mengajukan untuk mengundurkan diri dari pekerjaan sebagai dosen mulai hari ini karena saya baru saja mendapatkan jp paus yang cukup luar biasa dari situs MDG99 Gini gini pak, awalnya saya itu cuma sebagai iseng saja waktu lihat di iklan kayaknya seru gitu, ya udah dicoba dan saya sama istri pun kaget karena depo kecil aja malah bisa dapat puluhan juta Dan besoknya saya nulis surat kepada atasan saya bahwa saya mundur, ya saya udah tua juga sudah capek kerja untungnya juga ada situs MDG99 yang bisa bantu saya untuk kasih makan keluarga juga ya kan Saya tidak ada jam ngajar saya hari ini, sampai saya menulisnya selamanya Saya mundur mulai hari ini sampai dengan selamanya di dalam video.  Per tangkapan layar dibuat unggahan tersebut  sudah mendapatkan 68 komentar, dibagikan ulang 3 kali, dan disukai oleh 174 pengguna Facebook lainnya. "
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()

print("Label Prediksi:", pred)
print("Probabilities:", probs.tolist())
