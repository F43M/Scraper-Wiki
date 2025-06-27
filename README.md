# Scraper Wiki

Modular framework for scraping web content into machine learning datasets. It provides a CLI, plugin system and FastAPI server.

Full documentation is generated with `pdoc` and available under [docs/](docs/). Run:

```bash
pdoc -o docs -d google integrations core plugins utils api
```

See [docs/setup.md](docs/setup.md) for installation and configuration instructions and [docs/usage.md](docs/usage.md) for CLI and API examples.

## Installing Dependencies

Install the core packages first:

```bash
pip install -r requirements-core.txt
```

Machine learning features require an additional step:

```bash
pip install -r requirements-ml.txt
```

Workflow tools such as Airflow can be installed separately:

```bash
pip install -r requirements-workflow.txt
```

Additional deployment examples, including Kubernetes manifests and scaling guidance, are available in [docs/scaling.md](docs/scaling.md).
Scheduling options using Airflow or cron are covered in [docs/scheduling.md](docs/scheduling.md).
