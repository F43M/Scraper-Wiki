import json
from pathlib import Path
import typer

import scraper_wiki
import dashboard

app = typer.Typer(help="Scraper Wiki command line interface")

QUEUE_FILE = Path("jobs_queue.jsonl")

@app.command()
def scrape(
    lang: list[str] = typer.Option(None, "--lang", help="Idioma a processar", show_default=False),
    category: list[str] = typer.Option(None, "--category", help="Categoria específica", show_default=False),
    fmt: str = typer.Option("all", "--format", help="Formato de saída"),
):
    """Executa o scraper imediatamente."""
    scraper_wiki.main(lang, category, fmt)

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
    job = {"lang": lang, "category": category, "format": fmt}
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

if __name__ == "__main__":
    app()
