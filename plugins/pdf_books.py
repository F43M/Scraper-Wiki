"""Plugin to extract text from PDF books."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PyPDF2 import PdfReader

from .base import Plugin


class PDFBooksPlugin(Plugin):  # type: ignore[misc]
    """Parse PDF files in a directory or a single PDF."""

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return PDF file paths for a directory or single file."""
        path = Path(category)
        if path.is_file() and path.suffix.lower() == ".pdf":
            return [{"path": str(path), "lang": lang, "category": category}]
        if path.is_dir():
            pdfs = sorted(p for p in path.glob("*.pdf"))
            return [{"path": str(p), "lang": lang, "category": category} for p in pdfs]
        return []

    def parse_item(self, item: Dict) -> Dict:
        """Extract text from a PDF file."""
        file_path = item.get("path")
        if not file_path:
            return {}
        try:
            reader = PdfReader(file_path)
        except Exception:
            return {}
        text_parts: List[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        return {
            "path": file_path,
            "language": item.get("lang", "en"),
            "content": "\n".join(text_parts),
        }


# Registry alias
Plugin = PDFBooksPlugin
