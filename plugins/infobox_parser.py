"""Plugin to extract infobox data from Wikipedia pages."""
from typing import List, Dict

from bs4 import BeautifulSoup

from core import WikipediaAdvanced
from scraper_wiki import fetch_html_content

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
        soup = BeautifulSoup(html, "html.parser")
        box = soup.find("table", class_=lambda c: c and "infobox" in c)
        if not box:
            return {}
        data: Dict[str, str] = {}
        for row in box.find_all("tr"):
            header = row.find("th")
            value = row.find("td")
            if header and value:
                key = header.get_text(strip=True)
                val = value.get_text(strip=True)
                data[key] = val
        return {"title": item["title"], "language": item.get("lang", "en"), "infobox": data}
