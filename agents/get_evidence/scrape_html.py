import re
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv

load_dotenv()
SCRAPINGBEE_API_KEY = os.getenv("SCRAPINGBEE_API_KEY")

def scrape_html(url):
    try:
        api_url = "https://app.scrapingbee.com/api/v1/"
        params = {
            "api_key": SCRAPINGBEE_API_KEY,
            "url": url,
            "render_js": "true",
            "block_ads": "true",
            "wait": "4000"
        }

        response = requests.get(api_url, params=params, timeout=100)

        if response.status_code != 200:
            print("ScrapingBee error:", response.text)
            return None

        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        # ====== Judul ======
        judul = soup.title.string.strip() if soup.title and soup.title.string else None

        # ====== Tanggal ======
        time_tag = soup.find("time")
        tanggal = time_tag.get_text(strip=True) if time_tag else None

        # ====== Content ======
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        paragraphs = [p for p in paragraphs if len(p) > 50]

        content = "\n".join(paragraphs) if paragraphs else ""

        # ===========================================================
        #      VALIDATION: RETURN NONE JIKA KONTEN JELEK
        # ===========================================================

        # 1. Jika JS modal (khas detik)
        BAD_PATTERNS = [
            "aktifkan javascript",
            "upgrade web browser",
            "html5 video",
            "modal ini dapat ditutup",
        ]
        if any(bad in content.lower() for bad in BAD_PATTERNS):
            print("❌ halaman modal JS → gagal scrape")
            return None

        # 2. Konten terlalu pendek
        if len(content) < 200:
            print(f"❌ konten terlalu pendek ({len(content)} chars) → gagal scrape")
            return None

        # 3. Jika judul tidak ada
        if not judul:
            print("❌ tidak ada judul → gagal scrape")
            return None

        # ===========================================================
        #                     FEATURED IMAGE ONLY
        # ===========================================================

        featured_image = None

        # og:image
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            featured_image = urljoin(url, og["content"])

        # twitter:image
        if not featured_image:
            twitter = soup.find("meta", attrs={"name": "twitter:image"})
            if twitter and twitter.get("content"):
                featured_image = urljoin(url, twitter["content"])

        # JSON-LD
        if not featured_image:
            json_ld_tags = soup.find_all("script", type="application/ld+json")
            for tag in json_ld_tags:
                try:
                    import json
                    data = json.loads(tag.string)

                    if isinstance(data, dict):
                        image = data.get("image")
                        if isinstance(image, str):
                            featured_image = urljoin(url, image)
                            break
                        if isinstance(image, dict) and "url" in image:
                            featured_image = urljoin(url, image["url"])
                            break
                        if isinstance(image, list) and len(image) > 0:
                            featured_image = urljoin(url, image[0])
                            break
                except:
                    pass

        # Fallback gambar
        if not featured_image:
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src")
                if not src:
                    continue
                full_url = urljoin(url, src)
                if full_url.endswith((".svg", ".gif")):
                    continue
                featured_image = full_url
                break

        # ====== RETURN FINAL VALID DATA ======
        return {
            "judul": judul,
            "tanggal": tanggal or "Unknown",
            "sumber": urlparse(url).netloc.replace("www.", ""),
            "link": url,
            "content": content,
            "featured_image": featured_image,
        }

    except Exception as e:
        print(f"⚠️ Gagal scrape {url}: {e}")
        return None
