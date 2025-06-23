import os
from typing import List
from storage import StorageBackend


class DatalakeStorage(StorageBackend):
    """Store datasets in a local lake partitioned by ``lang`` and ``domain``."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def save_dataset(self, data: List[dict], fmt: str = "all") -> None:
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
