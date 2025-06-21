import re
from bs4 import BeautifulSoup


def clean_wiki_text(text: str) -> str:
    """Remove wiki markup and HTML from ``text``."""
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str, lang: str = "en") -> list[str]:
    """Split ``text`` into sentences using spaCy or fallback to NLTK."""
    try:
        import spacy  # type: ignore
        try:
            nlp = spacy.load(f"{lang}_core_web_sm")
        except Exception:
            nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [s.text.strip() for s in getattr(doc, "sents", []) if s.text.strip()]
    except Exception:
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            return [s.strip() for s in sent_tokenize(text, language=lang)]
        except Exception:
            return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
