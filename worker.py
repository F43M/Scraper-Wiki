"""Background worker that consumes page tasks and publishes results."""

import asyncio
import logging
from task_queue import consume, publish
from scraper_wiki import DatasetBuilder, Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main_sync() -> None:
    """Run worker in synchronous mode."""
    builder = DatasetBuilder()
    for task in consume("scrape_tasks"):
        logger.info("Processing %s", task.get("title"))
        result = builder.process_page(task)
        if result:
            publish("scrape_results", result)


async def _handle_task(
    task: dict, builder: DatasetBuilder, sem: asyncio.Semaphore
) -> None:
    async with sem:
        logger.info("Processing %s", task.get("title"))
        result = await builder.process_page_async(task)
        if result:
            await asyncio.to_thread(publish, "scrape_results", result)


async def main_async() -> None:
    """Run worker using asynchronous scraping."""
    builder = DatasetBuilder()
    sem = asyncio.Semaphore(Config.WORKER_CONCURRENCY)
    iterator = consume("scrape_tasks")
    while True:
        task = await asyncio.to_thread(next, iterator)
        asyncio.create_task(_handle_task(task, builder, sem))


def main(async_mode: bool = False) -> None:
    """Entry point for the worker."""
    if async_mode:
        asyncio.run(main_async())
    else:
        main_sync()


if __name__ == "__main__":
    import sys

    main(async_mode="--async" in sys.argv)
