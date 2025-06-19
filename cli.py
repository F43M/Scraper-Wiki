import json
import logging
from pathlib import Path
import typer

import scraper_wiki
import dashboard

app = typer.Typer(help="Scraper Wiki command line interface")


@app.callback(invoke_without_command=False)
def main(
    ctx: typer.Context,
    cache_backend: str = typer.Option(None, "--cache-backend", help="Backend de cache", show_default=False),
    cache_ttl: int = typer.Option(None, "--cache-ttl", help="Tempo de vida do cache em segundos", show_default=False),
    log_level: str = typer.Option(None, "--log-level", help="Nível de log (DEBUG, INFO, WARNING...)", show_default=False),
    log_format: str = typer.Option("text", "--log-format", help="Formato do log (text ou json)"),
    max_threads: int = typer.Option(None, "--max-threads", help="Número máximo de threads", show_default=False),
    max_processes: int = typer.Option(None, "--max-processes", help="Número máximo de processos", show_default=False),
    storage_backend: str = typer.Option(None, "--storage-backend", help="Backend de armazenamento", show_default=False),
):
    if cache_backend is not None:
        scraper_wiki.Config.CACHE_BACKEND = cache_backend
        scraper_wiki.cache = scraper_wiki.init_cache()
    if cache_ttl is not None:
        scraper_wiki.Config.CACHE_TTL = cache_ttl

    if log_level is not None or log_format != "text":
        level = getattr(logging, log_level.upper(), logging.INFO) if log_level else logging.INFO
        scraper_wiki.setup_logger("wiki_scraper", "scraper.log", level=level, fmt=log_format)
    if max_threads is not None:
        scraper_wiki.Config.MAX_THREADS = max_threads
    if max_processes is not None:
        scraper_wiki.Config.MAX_PROCESSES = max_processes
    if storage_backend is not None:
        scraper_wiki.Config.STORAGE_BACKEND = storage_backend

QUEUE_FILE = Path("jobs_queue.jsonl")

@app.command()
def scrape(
    lang: list[str] = typer.Option(None, "--lang", help="Idioma a processar", show_default=False),
    category: list[str] = typer.Option(None, "--category", help="Categoria específica", show_default=False),
    fmt: str = typer.Option("all", "--format", help="Formato de saída"),
    rate_limit_delay: float = typer.Option(None, "--rate-limit-delay", help="Delay entre requisições", show_default=False),
    plugin: str = typer.Option("wikipedia", "--plugin", help="Plugin de scraping"),
):
    """Executa o scraper imediatamente."""
    cats = [scraper_wiki.normalize_category(c) or c for c in category] if category else None
    if plugin == "wikipedia":
        scraper_wiki.main(lang, cats, fmt, rate_limit_delay)
    else:
        from plugins import load_plugin, run_plugin

        plg = load_plugin(plugin)
        languages = lang or scraper_wiki.Config.LANGUAGES
        categories = cats or list(scraper_wiki.Config.CATEGORIES)
        run_plugin(plg, languages, categories, fmt)

@app.command()
def monitor():
    """Inicia o dashboard para monitoramento."""
    dashboard.main()

@app.command()
def queue(
    lang: list[str] = typer.Option(None, "--lang", help="Idioma a processar", show_default=False),
    category: list[str] = typer.Option(None, "--category", help="Categoria específica", show_default=False),
    fmt: str = typer.Option("all", "--format", help="Formato de saída"),
):
    """Enfileira um job de scraping."""
    cats = [scraper_wiki.normalize_category(c) or c for c in category] if category else None
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

if __name__ == "__main__":
    app()
