import json
from pathlib import Path
from typing import List, Dict

from utils.text import clean_text


def save_hf_dataset(records: List[Dict], directory: Path) -> None:
    """Save ``records`` to a Hugging Face dataset directory.

    Parameters
    ----------
    records : list[dict]
        Dataset records.
    directory : pathlib.Path
        Target directory to store the dataset.
    """
    try:
        from datasets import Dataset

        ds = Dataset.from_list(records)
        directory.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(directory))
    except Exception:  # pragma: no cover - optional deps
        return


def save_tfrecord_dataset(records: List[Dict], path: Path) -> None:
    """Write ``records`` to a TFRecord file."""
    try:
        import tensorflow as tf

        path.parent.mkdir(parents=True, exist_ok=True)
        with tf.io.TFRecordWriter(str(path)) as writer:
            for rec in records:
                writer.write(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
    except Exception:  # pragma: no cover - optional deps
        return


def save_arrow_table(records: List[Dict], path: Path) -> None:
    """Export ``records`` to an Apache Arrow file."""
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc

        table = pa.Table.from_pylist(records)
        path.parent.mkdir(parents=True, exist_ok=True)
        with pa.OSFile(str(path), "wb") as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write(table)
    except Exception:  # pragma: no cover - optional deps
        return


def save_qa_dataset(records: List[Dict], path: Path) -> None:
    """Save question/answer pairs in a simple JSON list."""
    pairs = []
    for rec in records:
        qs = rec.get("questions", [])
        ans = rec.get("answers", [])
        for q, a in zip(qs, ans):
            q_text = q.get("text") if isinstance(q, dict) else q
            a_text = a.get("text") if isinstance(a, dict) else a
            pairs.append({"question": q_text, "answer": a_text})

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")


def save_text_corpus(records: List[Dict], directory: Path) -> None:
    """Write one cleaned ``.txt`` file per record."""
    directory.mkdir(parents=True, exist_ok=True)
    for rec in records:
        text = clean_text(rec.get("content", ""))
        name = rec.get("id") or rec.get("title", "record")
        file_path = directory / f"{name}.txt"
        file_path.write_text(text, encoding="utf-8")
