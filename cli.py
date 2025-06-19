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

if __name__ == "__main__":
    app()
