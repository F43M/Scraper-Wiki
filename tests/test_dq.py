import sys
from pathlib import Path

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dq


def test_deduplicate_by_simhash_removes_near_duplicates():
    records = [
        {"content": "Python is great for testing algorithms"},
        {"content": "Python is great for testing algorithm"},
        {"content": "Completely different text"},
    ]
    unique, removed = dq.deduplicate_by_simhash(records, distance=3)
    assert removed == 1
    assert len(unique) == 2


def test_complete_missing_fields_adds_new_keys():
    records = [
        {"title": "Python", "language": "en"},
    ]
    extra = [
        {"title": "Python", "language": "en", "wikidata_id": "Q1", "image_url": "img"}
    ]
    res = dq.complete_missing_fields(records, extra)
    assert res[0]["wikidata_id"] == "Q1"
    assert res[0]["image_url"] == "img"

