import os
import json
import csv
from typing import List

class StorageBackend:
    """Interface for storage backends."""
    def save_dataset(self, data: List[dict], fmt: str = 'all') -> None:
        raise NotImplementedError


class LocalStorage(StorageBackend):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_dataset(self, data: List[dict], fmt: str = 'all') -> None:
        if fmt in ['all', 'json']:
            json_file = os.path.join(self.output_dir, 'wikipedia_qa.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        if fmt in ['all', 'csv']:
            csv_file = os.path.join(self.output_dir, 'wikipedia_qa.csv')
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                rows = []
                for row in data:
                    converted = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()}
                    rows.append(converted)
                writer.writerows(rows)
        if fmt in ['all', 'parquet']:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                parquet_file = os.path.join(self.output_dir, 'wikipedia_qa.parquet')
                table = pa.Table.from_pylist(data)
                pq.write_table(table, parquet_file)
            except Exception:
                pass


class S3Storage(StorageBackend):
    def __init__(self, bucket: str, prefix: str = '', endpoint_url: str | None = None, client=None):
        try:
            import boto3
        except Exception as e:
            raise ImportError('boto3 is required for S3Storage') from e
        self.bucket = bucket
        self.prefix = prefix
        self.s3 = client or boto3.client('s3', endpoint_url=endpoint_url)

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def save_dataset(self, data: List[dict], fmt: str = 'all') -> None:
        if fmt in ['all', 'json']:
            body = json.dumps(data, ensure_ascii=False, indent=4).encode('utf-8')
            self.s3.put_object(Bucket=self.bucket, Key=self._key('wikipedia_qa.json'), Body=body)
        if fmt in ['all', 'csv']:
            import io
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=data[0].keys())
            writer.writeheader()
            for row in data:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
            self.s3.put_object(Bucket=self.bucket, Key=self._key('wikipedia_qa.csv'), Body=buffer.getvalue().encode('utf-8'))
        if fmt in ['all', 'parquet']:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.Table.from_pylist(data)
                buf = pa.BufferOutputStream()
                pq.write_table(table, buf)
                self.s3.put_object(Bucket=self.bucket, Key=self._key('wikipedia_qa.parquet'), Body=buf.getvalue().to_pybytes())
            except Exception:
                pass


class MongoDBStorage(StorageBackend):
    def __init__(self, uri: str, db_name: str = 'scraper', collection: str = 'dataset'):
        try:
            from pymongo import MongoClient
        except Exception as e:
            raise ImportError('pymongo is required for MongoDBStorage') from e
        client = MongoClient(uri)
        self.collection = client[db_name][collection]

    def save_dataset(self, data: List[dict], fmt: str = 'all') -> None:
        if not data:
            return
        self.collection.insert_many(data)


class PostgreSQLStorage(StorageBackend):
    def __init__(self, dsn: str, table: str = 'dataset'):
        try:
            import psycopg2
        except Exception as e:
            raise ImportError('psycopg2 is required for PostgreSQLStorage') from e
        self.conn = psycopg2.connect(dsn)
        self.table = table

    def save_dataset(self, data: List[dict], fmt: str = 'all') -> None:
        import json as _json
        with self.conn, self.conn.cursor() as cur:
            for row in data:
                cur.execute(f"INSERT INTO {self.table} (data) VALUES (%s)", [_json.dumps(row)])


def get_backend(name: str, output_dir: str):
    name = (name or 'local').lower()
    if name in ['s3', 'minio']:
        bucket = os.environ.get('S3_BUCKET', 'datasets')
        prefix = os.environ.get('S3_PREFIX', '')
        endpoint = os.environ.get('S3_ENDPOINT') if name == 's3' else os.environ.get('MINIO_ENDPOINT')
        return S3Storage(bucket, prefix=prefix, endpoint_url=endpoint)
    if name == 'mongodb':
        uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017')
        db = os.environ.get('MONGODB_DB', 'scraper')
        col = os.environ.get('MONGODB_COLLECTION', 'dataset')
        return MongoDBStorage(uri, db, col)
    if name == 'postgres':
        dsn = os.environ.get('POSTGRES_DSN', 'dbname=scraper user=postgres')
        table = os.environ.get('POSTGRES_TABLE', 'dataset')
        return PostgreSQLStorage(dsn, table)
    return LocalStorage(output_dir)
