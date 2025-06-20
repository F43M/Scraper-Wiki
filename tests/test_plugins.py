import importlib
import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scraper_wiki as sw
import core

class DummyWiki:
    def __init__(self, lang: str) -> None:
        self.lang = lang
    def get_category_members(self, category: str):
        return [{"title": "Page", "lang": self.lang, "category": category}]


def test_infobox_parser(monkeypatch):
    monkeypatch.setattr(core, "WikipediaAdvanced", DummyWiki)
    html = '<table class="infobox"><tr><th>Name</th><td>Python</td></tr></table>'
    monkeypatch.setattr(sw, "fetch_html_content", lambda title, lang: html)
    mod = importlib.reload(importlib.import_module("plugins.infobox_parser"))
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "Prog")
    assert items == [{"title": "Page", "lang": "en", "category": "Prog"}]
    parsed = plugin.parse_item(items[0])
    assert parsed == {"title": "Page", "language": "en", "infobox": {"Name": "Python"}}


def test_table_parser(monkeypatch):
    monkeypatch.setattr(core, "WikipediaAdvanced", DummyWiki)
    html = '<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>'
    monkeypatch.setattr(sw, "fetch_html_content", lambda title, lang: html)
    mod = importlib.reload(importlib.import_module("plugins.table_parser"))
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "Prog")
    assert items[0]["title"] == "Page"
    parsed = plugin.parse_item(items[0])
    assert parsed == {
        "title": "Page",
        "language": "en",
        "tables": [[["A", "B"], ["1", "2"]]],
    }

