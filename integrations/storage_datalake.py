import os
from pathlib import Path
from typing import List
from integrations.storage import StorageBackend


class DatalakeStorage(StorageBackend):
    """Store datasets in a local lake partitioned by ``lang`` and ``domain``."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def save_dataset(
        self,
        data: List[dict],
        fmt: str = "all",
        version: str | None = None,
        compression: str = "none",
    ) -> None:
        try:
            import pyarrow as pa
            import pyarrow.dataset as ds
        except Exception as e:  # pragma: no cover - missing deps
            raise ImportError("pyarrow is required for DatalakeStorage") from e

        if not data:
            return
        table = pa.Table.from_pylist(data)
        ds.write_dataset(
            table,
            base_dir=self.path,
            format="parquet",
            partitioning=["lang", "domain"],
            existing_data_behavior="overwrite_or_ignore",
        )
        if compression != "none":
            from utils.compression import compress_bytes
            import glob

            ext = ".zst" if compression == "zstd" else ".gz"
            for file in glob.glob(
                os.path.join(self.path, "**", "*.parquet"), recursive=True
            ):
                raw = Path(file).read_bytes()
                comp = compress_bytes(raw, compression)
                comp_path = file + ext
                Path(comp_path).write_bytes(comp)
                os.remove(file)
