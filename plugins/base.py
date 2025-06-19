"""Plugin interface definition."""
from typing import List, Dict, Protocol, runtime_checkable

@runtime_checkable
class Plugin(Protocol):
    """Basic scraping plugin."""

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return raw items for a given language and category."""
        ...

    def parse_item(self, item: Dict) -> Dict:
        """Convert a raw item into a dataset entry."""
        ...
