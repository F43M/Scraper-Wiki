"""Background worker that consumes page tasks and publishes results."""
import logging
from task_queue import consume, publish
from scraper_wiki import DatasetBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    builder = DatasetBuilder()
    for task in consume("scrape_tasks"):
        logger.info("Processing %s", task.get("title"))
        result = builder.process_page(task)
        if result:
            publish("scrape_results", result)


if __name__ == "__main__":
    main()
