"""Data quality utilities for deduplication and validation."""

from __future__ import annotations

import hashlib
from typing import List, Dict, Tuple

import numpy as np


def deduplicate_by_hash(records: List[Dict]) -> Tuple[List[Dict], int]:
    """Remove duplicate records using a hash of content.

    Parameters
    ----------
    records: List[Dict]
        Dataset records.

    Returns
    -------
    Tuple[List[Dict], int]
        Unique records and count of removed items.
    """
    seen: set[str] = set()
    unique: List[Dict] = []
    removed = 0

    for rec in records:
        base = rec.get("id") or f"{rec.get('title','')}_{rec.get('language','')}"
        if "content" in rec:
            base += rec["content"][:50]
        h = hashlib.md5(base.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(rec)
        else:
            removed += 1
    return unique, removed


def deduplicate_by_embedding(records: List[Dict], threshold: float = 0.95) -> Tuple[List[Dict], int]:
    """Remove semantically duplicated records using cosine similarity."""
    if not records:
        return records, 0

    embeddings = [rec.get("content_embedding", []) for rec in records]
    if not embeddings or not isinstance(embeddings[0], list):
        return records, 0

    emb = np.array(embeddings, dtype=float)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_emb = emb / norms

    to_remove = set()
    for i in range(len(records)):
        if i in to_remove:
            continue
        sims = np.dot(norm_emb[i], norm_emb[i + 1 :].T)
        for j, sim in enumerate(sims, start=i + 1):
            if sim >= threshold:
                to_remove.add(j)

    unique = [rec for idx, rec in enumerate(records) if idx not in to_remove]
    return unique, len(to_remove)


def validate_semantics(records: List[Dict]) -> Tuple[List[Dict], int]:
    """Validate semantic integrity of records."""
    valid: List[Dict] = []
    invalid = 0
    for rec in records:
        ok = True
        if not rec.get("content") or not rec.get("summary"):
            ok = False
        if not rec.get("questions") or not rec.get("answers"):
            ok = False
        if not np.all(np.isfinite(rec.get("content_embedding", []))):
            ok = False
        if not np.all(np.isfinite(rec.get("summary_embedding", []))):
            ok = False
        if ok:
            valid.append(rec)
        else:
            invalid += 1
    return valid, invalid


def complete_missing_fields(records: List[Dict], extra: List[Dict]) -> List[Dict]:
    """Fill empty fields using extra plugin data."""
    lookup = {
        (item.get("title"), item.get("language")): item for item in extra
    }
    for rec in records:
        key = (rec.get("title"), rec.get("language"))
        more = lookup.get(key)
        if not more:
            continue
        for k, v in more.items():
            if k not in rec or rec[k] in (None, "", []):
                rec[k] = v
    return records

