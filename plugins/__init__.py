"""Plugin utilities."""
import importlib
from typing import List
from core.builder import DatasetBuilder
from .base import Plugin


def load_plugin(name: str) -> Plugin:
    """Load a plugin by its module name."""
    module = importlib.import_module(f"plugins.{name}")
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
