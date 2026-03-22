import re
from urllib.parse import urlparse

def extract_title_from_url(url: str) -> str:
    path = urlparse(url).path
    slug = path.split("/")[-1]

    slug = re.sub(r'^\d+-\d+-', '', slug)
    title = slug.replace("-", " ")

    return title.lower()