import re
import unicodedata
from datetime import datetime
from typing import Dict

from bs4 import BeautifulSoup
import spacy

try:
    from dateutil import parser as date_parser
except Exception:  # pragma: no cover - optional dependency
    date_parser = None

nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    # Remove <sup> tags and their content
    soup = BeautifulSoup(text, "html.parser")
    for sup in soup.find_all("sup"):
        sup.decompose()
    text = soup.get_text()

    # Replace wiki style links [[Page|text]] -> text and [[Page]] -> Page
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^|\]]+)\]\]", r"\1", text)

    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_person(infobox: dict) -> dict:
    return {
        "name": infobox.get("name", ""),
        "birth_date": infobox.get("birth_date", ""),
        "occupation": infobox.get("occupation", "").split("|"),
    }


def parse_date(date_str: str) -> str:
    """Parse a date string into ISO 8601 format (YYYY-MM-DD)."""
    if not date_str:
        return ""
    if date_parser is not None:
        try:
            dt = date_parser.parse(date_str, fuzzy=True, dayfirst=False)
            return dt.date().isoformat()
        except Exception:
            return ""
    try:
        return datetime.fromisoformat(date_str).date().isoformat()
    except Exception:
        return ""


def normalize_infobox(infobox: Dict[str, str]) -> Dict[str, str]:
    """Return a normalized copy of an infobox.

    Keys containing the word "date" have their values parsed using
    :func:`parse_date` and all string fields are stripped.
    """
    normalized: Dict[str, str] = {}
    for key, value in infobox.items():
        val = value.strip() if isinstance(value, str) else value
        if "date" in key.lower():
            normalized[key] = parse_date(str(val))
        else:
            normalized[key] = val
    return normalized


def extract_entities(text: str) -> list[dict]:
    doc = nlp(text)
    return [{"text": ent.text, "type": ent.label_} for ent in doc.ents]
