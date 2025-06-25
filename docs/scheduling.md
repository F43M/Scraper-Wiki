# Scheduled Scraping

Scraper Wiki can run on a fixed schedule using either **Airflow** or a simple
cron job. The provided `training.airflow_pipeline` module builds an Airflow DAG
that collects pages, post-processes the dataset and publishes the results.
Set the `AIRFLOW_SCHEDULE` environment variable with a cron expression and start
Airflow using `docker-compose.airflow.yml`.

For lightweight deployments a cron job can execute the CLI directly. The
example below performs a weekly scrape every Monday at 02:00:

```cron
0 2 * * 1 python cli.py scrape --lang en --category "Programação" --format json
```

Both approaches track the last scraped timestamp of each page to avoid
redundant downloads and leverage DVC to store only the differences between
runs.

