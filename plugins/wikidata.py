"""Example Wikidata plugin."""
from .base import Plugin


class Plugin(Plugin):  # type: ignore[misc]
    def fetch_items(self, lang: str, category: str):
        # Placeholder implementation
        return []

    def parse_item(self, item: dict):
        # Placeholder implementation
        return {}

