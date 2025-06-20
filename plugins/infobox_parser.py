"""Plugin to extract infobox data from Wikipedia pages."""
from typing import List, Dict

from bs4 import BeautifulSoup

from core import WikipediaAdvanced
from scraper_wiki import fetch_html_content, extract_infobox

from .base import Plugin


class Plugin(Plugin):  # type: ignore[misc]
    """Extract infobox information."""

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        wiki = WikipediaAdvanced(lang)
        return wiki.get_category_members(category)

    def parse_item(self, item: Dict) -> Dict:
        html = fetch_html_content(item["title"], item.get("lang", "en"))
        if not html:
            return {}
        data = extract_infobox(html)
        if not data:
            return {}
        return {
            "title": item["title"],
            "language": item.get("lang", "en"),
            "infobox": data,
        }
