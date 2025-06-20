"""StackOverflow scraping plugin."""

from typing import List, Dict

import html2text
import requests

from scraper_wiki import Config, advanced_clean_text
from .base import Plugin


class Plugin(Plugin):  # type: ignore[misc]
    """Fetch questions from StackOverflow by tag."""

    def __init__(self, api_key: str | None = None, endpoint: str | None = None) -> None:
        self.api_key = api_key or Config.STACKOVERFLOW_API_KEY
        self.endpoint = endpoint or Config.STACKOVERFLOW_API_ENDPOINT.rstrip("/")

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        params = {
            "site": "stackoverflow",
            "tagged": category,
            "pagesize": 10,
            "order": "desc",
            "sort": "votes",
            "filter": "withbody",
        }
        if self.api_key:
            params["key"] = self.api_key
        resp = requests.get(f"{self.endpoint}/questions", params=params, timeout=Config.TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        for it in items:
            it["lang"] = lang
            it["category"] = category
        return items

    def parse_item(self, item: Dict) -> Dict:
        body = item.get("body", "")
        text = html2text.html2text(body) if hasattr(html2text, "html2text") else body
        clean = advanced_clean_text(text, item.get("lang", "en"))
        return {
            "title": item.get("title", ""),
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "score": item.get("score", 0),
            "link": item.get("link", ""),
            "content": clean,
        }

