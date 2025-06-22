import importlib
import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sw = importlib.import_module("scraper_wiki")


def test_generate_qa_pairs_processes_code(monkeypatch):
    builder = sw.DatasetBuilder()
    monkeypatch.setattr(builder, "_generate_questions", lambda *a, **k: [])
    monkeypatch.setattr(builder, "_generate_answers", lambda *a, **k: [])
    monkeypatch.setattr(sw, "extract_relations", lambda *a, **k: [])
    import numpy as np

    monkeypatch.setattr(
        builder.embedding_model, "encode", lambda *a, **k: np.array([0.0])
    )
    code = "def foo():\n    pass  # comment"
    result = builder.generate_qa_pairs("T", code, "S", "en", "c")
    assert result["content"] == "def foo():\n    pass"
    assert result["metadata"].get("code_language") == "python"
