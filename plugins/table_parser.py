"""Plugin to extract table data from Wikipedia pages."""
from typing import List, Dict

from bs4 import BeautifulSoup

from core import WikipediaAdvanced
from scraper_wiki import fetch_html_content

from .base import Plugin


class Plugin(Plugin):  # type: ignore[misc]
    """Parse all HTML tables on a page."""

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        wiki = WikipediaAdvanced(lang)
        return wiki.get_category_members(category)

    def parse_item(self, item: Dict) -> Dict:
        html = fetch_html_content(item["title"], item.get("lang", "en"))
        if not html:
            return {}
        soup = BeautifulSoup(html, "html.parser")
        tables: List[List[List[str]]] = []
        for table in soup.find_all("table"):
            rows: List[List[str]] = []
            for tr in table.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                if cells:
                    rows.append(cells)
            if rows:
                tables.append(rows)
        if not tables:
            return {}
        return {"title": item["title"], "language": item.get("lang", "en"), "tables": tables}
