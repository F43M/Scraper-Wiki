"""Plugin utilities and registry."""
import importlib
from typing import List
from core.builder import DatasetBuilder
from .base import Plugin

# Mapping of available plugin names to module paths
AVAILABLE_PLUGINS = {
    "wikipedia": "wikipedia",
    "wikidata": "wikidata",
    "stackoverflow": "stackoverflow",
    "infobox_parser": "infobox_parser",
    "table_parser": "table_parser",
}


def load_plugin(name: str) -> Plugin:
    """Load a plugin by its registry name."""
    module_name = AVAILABLE_PLUGINS.get(name, name)
    module = importlib.import_module(f"plugins.{module_name}")
    plugin_cls = getattr(module, "Plugin")
    return plugin_cls()


def run_plugin(plugin: Plugin, langs: List[str], categories: List[str], fmt: str = "all") -> List[dict]:
    """Execute scraping using a plugin."""
    builder = DatasetBuilder()
    for lang in langs:
        for category in categories:
            items = plugin.fetch_items(lang, category)
            for item in items:
                result = plugin.parse_item(item)
                if result:
                    builder.dataset.append(result)
    builder.save_dataset(fmt)
    return builder.dataset

__all__ = ["load_plugin", "run_plugin", "AVAILABLE_PLUGINS"]
