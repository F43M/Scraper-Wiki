"""Programming competition scraping plugin."""

from __future__ import annotations

from typing import List, Dict

import requests

from scraper_wiki import Config
from .base import Plugin


class CompetitionsPlugin(Plugin):  # type: ignore[misc]
    """Fetch problems and solutions from LeetCode and CodeWars."""

    def __init__(
        self, leetcode_url: str | None = None, codewars_url: str | None = None
    ) -> None:
        self.leetcode_url = leetcode_url or "https://leetcode.com/api/problems"
        self.codewars_url = codewars_url or "https://www.codewars.com/api/v1"

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return basic problem identifiers for both platforms."""
        return [
            {"source": "leetcode", "slug": category, "lang": lang},
            {"source": "codewars", "slug": category, "lang": lang},
        ]

    def _fetch_leetcode(self, slug: str) -> Dict:
        resp = requests.get(f"{self.leetcode_url}/{slug}", timeout=Config.TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def _fetch_codewars(self, slug: str) -> Dict:
        resp = requests.get(f"{self.codewars_url}/kata/{slug}", timeout=Config.TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def parse_item(self, item: Dict) -> Dict:
        """Return problem statement and accepted solution."""
        source = item.get("source")
        slug = item.get("slug")
        if source == "leetcode":
            data = self._fetch_leetcode(slug)
            return {
                "problem": data.get("content", ""),
                "solution": data.get("solution", ""),
            }
        if source == "codewars":
            data = self._fetch_codewars(slug)
            solutions = data.get("solutions", [])
            solution = solutions[0] if solutions else ""
            return {"problem": data.get("description", ""), "solution": solution}
        return {}


# Registry alias
Plugin = CompetitionsPlugin
