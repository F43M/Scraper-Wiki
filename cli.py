import json
import logging
from pathlib import Path
from typing import List, Optional

import typer

import scraper_wiki
import dashboard
from search import indexer

app = typer.Typer(help="Scraper Wiki command line interface")


@app.callback(invoke_without_command=False)
def main(
    ctx: typer.Context,
    cache_backend: str = typer.Option(
        None, "--cache-backend", help="Backend de cache", show_default=False
    ),
    cache_ttl: int = typer.Option(
        None,
        "--cache-ttl",
        help="Tempo de vida do cache em segundos",
        show_default=False,
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        help="Nível de log (DEBUG, INFO, WARNING...)",
        show_default=False,
    ),
    log_format: str = typer.Option(
        "text", "--log-format", help="Formato do log (text ou json)"
    ),
    max_threads: int = typer.Option(
        None, "--max-threads", help="Número máximo de threads", show_default=False
    ),
    max_processes: int = typer.Option(
        None, "--max-processes", help="Número máximo de processos", show_default=False
    ),
    storage_backend: str = typer.Option(
        None, "--storage-backend", help="Backend de armazenamento", show_default=False
    ),
):
    if cache_backend is not None:
        scraper_wiki.Config.CACHE_BACKEND = cache_backend
        scraper_wiki.cache = scraper_wiki.init_cache()
    if cache_ttl is not None:
        scraper_wiki.Config.CACHE_TTL = cache_ttl

    if log_level is not None or log_format != "text":
        level = (
            getattr(logging, log_level.upper(), logging.INFO)
            if log_level
            else logging.INFO
        )
        scraper_wiki.setup_logger(
            "wiki_scraper", "scraper.log", level=level, fmt=log_format
        )
    if max_threads is not None:
        scraper_wiki.Config.MAX_THREADS = max_threads
    if max_processes is not None:
        scraper_wiki.Config.MAX_PROCESSES = max_processes
    if storage_backend is not None:
        scraper_wiki.Config.STORAGE_BACKEND = storage_backend


QUEUE_FILE = Path("jobs_queue.jsonl")


@app.command()
def scrape(
    lang: Optional[List[str]] = typer.Option(
        None, "--lang", help="Idioma a processar", show_default=False
    ),
    category: Optional[List[str]] = typer.Option(
        None, "--category", help="Categoria específica", show_default=False
    ),
    fmt: str = typer.Option(
        "all",
        "--format",
        help="Formato de saída (json, jsonl, csv, parquet, tfrecord, qa, text)",
    ),
    start_page: list[str] = typer.Option(
        None,
        "--start-page",
        help="Página inicial para rastrear links (pode ser repetido)",
        show_default=False,
    ),
    depth: int = typer.Option(
        1,
        "--depth",
        help="Profundidade de navegação para páginas iniciais",
    ),
    rate_limit_delay: float = typer.Option(
        None, "--rate-limit-delay", help="Delay entre requisições", show_default=False
    ),
    revisions: bool = typer.Option(
        False, "--revisions", help="Inclui histórico de revisões", is_flag=True
    ),
    rev_limit: int = typer.Option(5, "--rev-limit", help="Número máximo de revisões"),
    async_mode: bool = typer.Option(
        False, "--async", help="Usa scraping assíncrono", is_flag=True
    ),
    plugin: str = typer.Option(
        "wikipedia",
        "--plugin",
        help="Plugin de scraping (wikipedia, infobox_parser, table_parser)",
    ),
    distributed: bool = typer.Option(
        False, "--distributed", help="Usa cluster distribuído", is_flag=True
    ),
    train: bool = typer.Option(
        False, "--train", help="Executa conversões para treinamento"
    ),
    incremental: bool = typer.Option(
        False, "--incremental", help="Busca apenas novos itens", is_flag=True
    ),
):
    """Executa o scraper imediatamente."""
    lang = lang or None
    category = category or None
    cats = (
        [scraper_wiki.normalize_category(c) or c for c in category]
        if category
        else None
    )
    client = None
    if distributed:
        from cluster import get_client

        client = get_client()

    if plugin == "wikipedia":
        if async_mode:
            import asyncio

            asyncio.run(
                scraper_wiki.main_async(
                    lang,
                    cats,
                    fmt,
                    rate_limit_delay,
                    start_pages=start_page,
                    depth=depth,
                    revisions=revisions,
                    rev_limit=rev_limit,
                )
            )
        else:
            scraper_wiki.main(
                lang,
                cats,
                fmt,
                rate_limit_delay,
                start_pages=start_page,
                depth=depth,
                revisions=revisions,
                rev_limit=rev_limit,
                client=client,
            )
        dataset_file = Path(scraper_wiki.Config.OUTPUT_DIR) / "wikipedia_qa.json"
        if dataset_file.exists() and train:
            from training import pipeline

            pipeline.run_pipeline(dataset_file)
    else:
        from plugins import load_plugin, run_plugin

        plg = load_plugin(plugin)
        languages = lang or scraper_wiki.Config.LANGUAGES
        categories = cats or list(scraper_wiki.Config.CATEGORIES)
        run_plugin(plg, languages, categories, fmt, incremental=incremental)


@app.command()
def monitor():
    """Inicia o dashboard para monitoramento."""
    dashboard.main()


@app.command()
def queue(
    lang: Optional[List[str]] = typer.Option(
        None, "--lang", help="Idioma a processar", show_default=False
    ),
    category: Optional[List[str]] = typer.Option(
        None, "--category", help="Categoria específica", show_default=False
    ),
    fmt: str = typer.Option(
        "all",
        "--format",
        help="Formato de saída (json, jsonl, csv, parquet, tfrecord, qa, text)",
    ),
):
    """Enfileira um job de scraping."""
    lang = lang or None
    category = category or None
    cats = (
        [scraper_wiki.normalize_category(c) or c for c in category]
        if category
        else None
    )
    job = {"lang": lang, "category": cats, "format": fmt}
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(job, ensure_ascii=False) + "\n")
    typer.echo(f"Job enfileirado: {job}")


@app.command()
def status():
    """Mostra arquivos gerados e configurações principais."""
    from scraper_wiki import Config

    output_dir = Path(Config.OUTPUT_DIR)
    typer.echo(f"Conteúdo de {output_dir}:")
    if output_dir.exists():
        for path in output_dir.iterdir():
            typer.echo(f"- {path.name}")
    else:
        typer.echo("(diretório não encontrado)")

    typer.echo("\nConfigurações chave:")
    settings = {
        "OUTPUT_DIR": Config.OUTPUT_DIR,
        "CACHE_DIR": Config.CACHE_DIR,
        "LOG_DIR": Config.LOG_DIR,
        "MAX_THREADS": Config.MAX_THREADS,
        "MAX_PROCESSES": Config.MAX_PROCESSES,
    }
    for key, value in settings.items():
        typer.echo(f"{key}: {value}")


@app.command("clear-cache")
def clear_cache_cmd():
    """Remove entradas expiradas do cache."""
    scraper_wiki.clear_cache()
    typer.echo("Cache limpo")


@app.command("search")
def search_cli(query: str):
    """Search indexed records using Elasticsearch."""
    results = indexer.query_index(query)
    typer.echo(json.dumps(results, ensure_ascii=False))


@app.command("process")
def process_pipeline(
    dataset: str = typer.Argument(..., help="Caminho do dataset"),
    pipeline: str = typer.Option("default", "--pipeline", help="Nome do pipeline"),
):
    """Run processing pipeline on an existing dataset."""
    from processing.pipeline import get_pipeline

    data_path = Path(dataset)
    records = json.loads(data_path.read_text(encoding="utf-8"))
    pipe = get_pipeline(pipeline)
    result = pipe(records)
    data_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    typer.echo(f"Processados {len(result)} registros")


@app.command("start-crawler")
def start_crawler_cmd(
    config: str = typer.Option(None, "--config", help="Path to cluster config")
):
    """Start the distributed crawler."""

    from crawling.distributed import start_crawler

    start_crawler(config)


@app.command("stop-crawler")
def stop_crawler_cmd():
    """Stop the distributed crawler."""

    from crawling.distributed import stop_crawler

    stop_crawler()


if __name__ == "__main__":
    app()
