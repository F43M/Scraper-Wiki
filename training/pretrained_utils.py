"""Utilities for working with pretrained models.

This module provides helper functions to prepare inputs for BERT models and to
build simple image datasets compatible with Stable Diffusion training.

Example
-------
>>> from training.pretrained_utils import prepare_bert_inputs
>>> inputs = prepare_bert_inputs(["Hello World"])  # doctest: +SKIP
>>> inputs["input_ids"].shape[0] == 1
True
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import requests
import torch
from transformers import BertTokenizer


def prepare_bert_inputs(texts: List[str]) -> Dict[str, torch.Tensor]:
    """Tokenize ``texts`` with ``BertTokenizer`` and return tensors.

    Parameters
    ----------
    texts : list[str]
        Sentences or documents to encode.

    Returns
    -------
    Dict[str, torch.Tensor]
        Tokenized inputs ready for ``torch`` models.

    Example
    -------
    >>> from training.pretrained_utils import prepare_bert_inputs
    >>> res = prepare_bert_inputs(["example text"])  # doctest: +SKIP
    >>> list(res.keys())
    ['input_ids', 'token_type_ids', 'attention_mask']
    """

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


def extract_image_dataset(records: List[dict], out_dir: Path) -> None:
    """Download image URLs and captions for Stable Diffusion.

    Each record should contain ``image_url`` and ``caption`` or ``title`` fields.
    Images are saved inside ``out_dir`` using zero-padded names and a
    ``captions.txt`` file maps filenames to captions.

    Example
    -------
    >>> recs = [{"image_url": "http://example.com/pic.jpg", "caption": "A cat"}]
    >>> extract_image_dataset(recs, Path('data'))  # doctest: +SKIP
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    captions = out_dir / "captions.txt"
    with captions.open("w", encoding="utf-8") as cf:
        for idx, rec in enumerate(records):
            url = rec.get("image_url")
            if not url:
                continue
            caption = rec.get("caption") or rec.get("title", "")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            ext = Path(url).suffix or ".jpg"
            name = f"{idx:05d}{ext}"
            with open(out_dir / name, "wb") as img_f:
                img_f.write(resp.content)
            cf.write(f"{name}\t{caption}\n")
