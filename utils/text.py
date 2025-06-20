import re
import spacy

nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_person(infobox: dict) -> dict:
    return {
        "name": infobox.get("name", ""),
        "birth_date": infobox.get("birth_date", ""),
        "occupation": infobox.get("occupation", "").split("|"),
    }


def extract_entities(text: str) -> list[dict]:
    doc = nlp(text)
    return [{"text": ent.text, "type": ent.label_} for ent in doc.ents]
