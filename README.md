# Scraper Wiki

Modular framework for scraping web content into machine learning datasets. It provides a CLI, plugin system and FastAPI server.

Full documentation is generated with `pdoc` and available under [docs/](docs/). Run:

```bash
pdoc -o docs -d google integrations core plugins utils api
```

See [docs/setup.md](docs/setup.md) for installation and configuration instructions and [docs/usage.md](docs/usage.md) for CLI and API examples.

Additional deployment examples, including Kubernetes manifests and scaling guidance, are available in [docs/scaling.md](docs/scaling.md).
Scheduling options using Airflow or cron are covered in [docs/scheduling.md](docs/scheduling.md).

## Starting the Interface

Launch the API server with:

```bash
python cli.py serve
```

By default it listens on `http://0.0.0.0:8000`. A typical workflow is to start
the server and then issue requests such as `curl http://localhost:8000/api/plugins`
or POST jobs to `/api/scrape` as shown in [docs/usage.md](docs/usage.md).
