import json
from pathlib import Path
from typing import List, Dict


def load_dataset(path: str | Path) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def convert_to_conversation_pairs(records: List[Dict]) -> List[Dict]:
    """Return conversation pairs from dataset."""
    pairs = []
    for rec in records:
        qs = rec.get("questions", [])
        ans = rec.get("answers", [])
        for q, a in zip(qs, ans):
            q_text = q["text"] if isinstance(q, dict) else q
            a_text = a["text"] if isinstance(a, dict) else a
            pairs.append({"question": q_text, "answer": a_text})
    return pairs


def convert_to_embeddings(records: List[Dict]) -> List[Dict]:
    """Extract embeddings from dataset."""
    emb = []
    for rec in records:
        emb.append({
            "id": rec.get("id"),
            "embedding": rec.get("content_embedding")
        })
    return emb


def convert_to_triples(records: List[Dict]) -> List[Dict]:
    """Generate knowledge triples using keywords."""
    triples = []
    for rec in records:
        subj = rec.get("title")
        for kw in rec.get("keywords", []):
            triples.append({"subject": subj, "relation": "related_to", "object": kw})
    return triples


def run_pipeline(dataset_path: str | Path) -> None:
    """Run full conversion pipeline for training."""
    dataset_path = Path(dataset_path)
    records = load_dataset(dataset_path)
    base = dataset_path.with_suffix("")

    pairs = convert_to_conversation_pairs(records)
    save_json(pairs, base.with_name(base.name + "_pairs.json"))

    emb = convert_to_embeddings(records)
    save_json(emb, base.with_name(base.name + "_embeddings.json"))

    triples = convert_to_triples(records)
    save_json(triples, base.with_name(base.name + "_triples.json"))
