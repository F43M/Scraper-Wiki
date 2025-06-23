"""Data processing pipeline stages."""

from __future__ import annotations

import json
import subprocess
import re
from typing import Callable, Dict, List

import dq
from training.postprocessing import analyze_code_ast


def normalize_encoding(text: str) -> str:
    """Return ``text`` re-encoded as UTF-8, dropping invalid sequences."""
    if isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


def tokenize(text: str) -> List[str]:
    """Tokenize ``text`` preserving punctuation."""
    return re.findall(r"\w+|\S", text)


def lint_code(code: str) -> List[str]:
    """Run flake8 on ``code`` returning warnings."""
    try:
        res = subprocess.run(
            ["flake8", "-"],
            input=code,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    out = (res.stdout + res.stderr).strip()
    return [l for l in out.splitlines() if l]


def scan_vulnerabilities(code: str) -> List[str]:
    """Scan ``code`` using semgrep if available."""
    try:
        result = subprocess.run(
            ["semgrep", "--quiet", "--config=p/ci", "-"],
            input=code,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    output = result.stdout.strip()
    return [l for l in output.splitlines() if l]


def analyze_complexity(code: str, language: str | None = None) -> Dict[str, int]:
    """Return complexity metrics for ``code`` without altering it."""
    analysis = analyze_code_ast(code, language)
    return analysis.get("complexities", {})


def process_record(record: Dict) -> Dict:
    """Process a single dataset record through all stages."""
    content = record.get("content", "")
    content = normalize_encoding(content)
    record["content"] = content
    record["tokens"] = tokenize(content)
    record["lint"] = lint_code(content)
    record["vulnerabilities"] = scan_vulnerabilities(content)
    comp = analyze_complexity(content, record.get("metadata", {}).get("code_language"))
    if comp:
        record.setdefault("metadata", {})["complexities"] = comp
    return record


def run_pipeline(records: List[Dict]) -> List[Dict]:
    """Run the full processing pipeline on ``records``."""
    processed = [process_record(rec) for rec in records]
    processed, _ = dq.deduplicate_by_embedding(processed)
    return processed


def get_pipeline(name: str = "default") -> Callable[[List[Dict]], List[Dict]]:
    """Return pipeline callable by ``name``."""
    if name == "default":
        return run_pipeline
    raise ValueError(f"Unknown pipeline: {name}")


__all__ = [
    "normalize_encoding",
    "tokenize",
    "lint_code",
    "scan_vulnerabilities",
    "analyze_complexity",
    "process_record",
    "run_pipeline",
    "get_pipeline",
]
