import importlib
import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

code_mod = importlib.import_module("utils.code")


def test_normalize_indentation():
    raw = "    def f():\n        pass\n"
    assert code_mod.normalize_indentation(raw) == "def f():\n    pass"


def test_remove_comments_python():
    raw = "def f():\n    # c\n    return 1"
    assert code_mod.remove_comments(raw, "python") == "def f():\n    return 1"


def test_detect_programming_language():
    assert code_mod.detect_programming_language("def f():\n    pass") == "python"
    assert code_mod.detect_programming_language("console.log('x');") == "javascript"
