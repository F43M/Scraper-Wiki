"""Minimal search web UI using plugins."""

from typing import Any

from plugins import load_plugin, run_plugin


def execute_search(term: str, plugin_name: str = "wikipedia") -> list[dict[str, Any]]:
    """Run ``plugin_name`` for ``term`` and return collected records."""
    plugin = load_plugin(plugin_name)
    return run_plugin(plugin, langs=["en"], categories=[term], fmt="json")


if __name__ == "__main__":  # pragma: no cover - manual execution
    import argparse

    parser = argparse.ArgumentParser(description="Run a plugin search")
    parser.add_argument("term", help="Search term")
    parser.add_argument("--plugin", default="wikipedia", help="Plugin name")
    args = parser.parse_args()
    execute_search(args.term, args.plugin)
