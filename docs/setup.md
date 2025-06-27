# Setup

## Installation

```bash
pip install -r requirements-core.txt
```

For machine learning features install the second set:

```bash
pip install -r requirements-ml.txt
```

Tools like Airflow are provided separately:

```bash
pip install -r requirements-workflow.txt
```

## Environment Variables

- `STORAGE_BACKEND`: `local`, `s3`, `mongodb`, `postgres`, `iceberg`, `delta`, `datalake`, `neo4j`, `milvus`, or `weaviate`.
- `S3_BUCKET`, `S3_ENDPOINT`: Configure S3/MinIO storage.
- `MONGODB_URI`: MongoDB connection string.
- `POSTGRES_DSN`: PostgreSQL DSN.
- `DATALAKE_PATH`: Path for Iceberg/Delta Lake.
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Neo4j connection.
- `MILVUS_URI`, `MILVUS_COLLECTION`: Milvus vector store.
- `WEAVIATE_URI`: Weaviate endpoint.
- `USE_GPU`: Set to `1` or `true` to force GPU usage, `0` to disable. When unset,
  GPU availability is detected automatically via `torch.cuda.is_available()`.
- `API_TOKEN`: Optional bearer token required by the FastAPI server. If unset
  all requests are accepted without authentication.

