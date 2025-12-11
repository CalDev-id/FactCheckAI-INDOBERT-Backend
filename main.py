import json
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from agents.predict.predict import classify_berita
from agents.predict.predict import advance_classify_berita
from agents.get_evidence.google_search import google_search
from agents.get_evidence.scrape_html import scrape_html
from agents.explanation.explanation import explanation
from agents.chat.chat import agent
from auth.supabase_client import supabase
from typing import List, Any, Optional

#uvicorn main:app --reload
#uvicorn main:app --host 0.0.0.0 --port 8000
#hcsp_1_5g
#http://192.168.50.110:8000/docs

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Fake News Detection API"}

class PredictRequest(BaseModel):
    title: str
    content: str

@app.post("/predict/")
def predict_fake_news(data: PredictRequest):
    result = classify_berita(data.title, data.content)
    return result

class ClaimRequest(BaseModel):
    claim: str

@app.post("/get_evidence/")
def get_evidence(
    data: ClaimRequest,
):
    total_results: int = 10
    scrape_limit: int = 1  # cuma mau 1 evidence
    query = data.claim
    # 1. Google Search
    links = google_search(query, total_results=total_results)

    # 2. Scraping dengan fallback ke link berikutnya
    scraped = []

    for url in links:  # cek semua link
        if len(scraped) >= scrape_limit:
            break  # sudah cukup evidence

        content = scrape_html(url)

        if content is None:
            print(f"❌ Gagal scrape → {url}")
            continue  # coba link berikutnya

        print(f"✅ Berhasil scrape → {url}")
        scraped.append(content)

    return {
        "query": query,
        "total_google_results": len(links),
        "scraped_results": len(scraped),
        "links": links,
        "evidence": scraped
    }


@app.post("/predict_with_evidence/")
def predict_with_evidence(data: PredictRequest):
    total_results = 10
    scrape_limit = 1
    links = google_search(data.title, total_results=total_results)

    scraped = []
    for url in links:
        if len(scraped) >= scrape_limit:
            break
        content = scrape_html(url)
        if content is None:
            print(f"❌ Gagal scrape → {url}")
            continue
        print(f"✅ Berhasil scrape → {url}")
        scraped.append(content)

    classification = classify_berita(data.title, data.content)

    advance_classification = advance_classify_berita(
        classification=classification,
        news_scrape=scraped,
        title=data.title,
        evidence_link=links,
        content=content
    )

    llm_output = explanation(
        classification=classification,
        news_scrape=scraped,
        evidence_link=links,
        title=data.title,
        content=data.content
    )

    return {
        "url": scraped[0].get("link", "") if scraped else "",
        "title": data.title,
        "content": data.content,
        "classification": advance_classification,
        "evidence_links": links,
        "evidence_scraped": scraped,
        "explanation": llm_output 
    }

class UrlRequest(BaseModel):
    url: str
    
@app.post("/predict_from_url/")
def predict_from_url(data: UrlRequest):

    # 1. Scrape artikel dari URL input
    scraped_main = scrape_html(data.url)
    if not scraped_main or scraped_main.get("content") == "Tidak berhasil ekstrak isi artikel":
        return {
            "error": "Gagal mengambil artikel dari URL",
            "url": data.url
        }

    title = scraped_main.get("judul", "")
    content = scraped_main.get("content", "")

    # 2. Klasifikasi IndoBERT
    classification = classify_berita(title, content)

    # 3. Google Search
    total_results = 10
    scrape_limit = 1
    links = google_search(title, total_results=total_results)

    # 4. Scrape evidence
    scraped_evidence = []
    for url in links:  # cek semua link
        if len(scraped_evidence) >= scrape_limit:
            break  # sudah cukup evidence

        content = scrape_html(url)

        if content is None:
            print(f"❌ Gagal scrape → {url}")
            continue  # coba link berikutnya

        print(f"✅ Berhasil scrape → {url}")
        scraped_evidence.append(content)

    # 5. Advance Classification
    advance_classification = advance_classify_berita(
        classification=classification,
        news_scrape=scraped_evidence,
        title=title,
        evidence_link=links,
        content=content
    )

    # 6. LLM judgement
    llm_output = explanation(
        classification=advance_classification,
        news_scrape=scraped_evidence,
        title=title,
        evidence_link=links,
        content=content
    )

    # 7. Output final
    return {
        "url": data.url,
        "title": title,
        "content": content,
        "classification": advance_classification,
        "evidence_links": links,
        "evidence_scraped": scraped_evidence,
        "explanation": llm_output
    }

    
@app.post("/predict_from_claim/")
def predict_from_claim(data: ClaimRequest):

    total_results: int = 10
    scrape_limit: int = 1
    query = data.claim

    links = google_search(query, total_results=total_results)

    scraped = []
    for url in links:
        if len(scraped) >= scrape_limit:
            break
        content = scrape_html(url)
        if content is None:
            print(f"❌ Gagal scrape → {url}")
            continue
        print(f"✅ Berhasil scrape → {url}")
        scraped.append(content)

    classification = classify_berita(scraped[0].get("judul", ""), scraped[0].get("content", ""))

    advance_classification = advance_classify_berita(
        classification=classification,
        news_scrape=scraped,
        title=scraped[0].get("judul", ""),
        evidence_link=links,
        content=scraped[0].get("content", "")
    )

    llm_output = explanation(
        classification=advance_classification,
        news_scrape=scraped,
        title=scraped[0].get("judul", ""),
        evidence_link=links,
        content=scraped[0].get("content", "")
    )

    return {
        "url": links[0] if links else "",
        "title": scraped[0].get("judul", "") if scraped else "",
        "content": scraped[0].get("content", "") if scraped else "",
        "classification": advance_classification,
        "evidence_links": links,
        "evidence_scraped": scraped,
        "explanation": llm_output
    }


# CHAT AGENT
class ChatRequest(BaseModel):
    message: str

@app.post("/chat/")
def chat_endpoint(data: ChatRequest):
    user_message = data.message

    try:
        response = agent.run(user_message)
        return {
            "response": response
        }
    except Exception as e:
        return {"error": str(e)}
    

# SUPABASE

@app.get("/news")
def get_news():
    result = supabase.table("news").select("*").execute()
    return result.data

@app.get("/news/hoaks")
def get_hoax_news():
    result = supabase.table("news").select("*").eq("classification", "hoaks").execute()
    return result.data

@app.get("/news/valid")
def get_valid_news():
    result = supabase.table("news").select("*").eq("classification", "valid").execute()
    return result.data

# TARUH PALING BAWAH
from uuid import UUID
@app.get("/news/{news_id}")
def get_news_by_id(news_id: UUID):
    result = supabase.table("news").select("*").eq("id", str(news_id)).single().execute()
    return result.data


@app.get("/news/search")
def search_news(q: str):
    result = supabase.table("news").select("*").ilike("title", f"%{q}%").execute()
    return result.data

class NewsPayload(BaseModel):
    url: str
    title: str
    content: str
    classification: str
    evidence_link: Optional[List[str]] = None
    evidence_scraped: Optional[Any] = None
    explanation: Optional[str] = None


@app.post("/news")
def insert_news(payload: NewsPayload):
    data = payload.dict()
    result = supabase.table("news").insert(data).execute()
    return result.data