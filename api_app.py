from typing import List, Optional
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import scraper_wiki as sw
import os
import json
from datetime import datetime
import graphene
from utils.text import clean_text, extract_entities
import asyncio

app = FastAPI()

DATA_FILE = os.path.join(sw.Config.OUTPUT_DIR, "wikipedia_qa.json")
PROGRESS_FILE = os.path.join(sw.Config.LOG_DIR, "progress.json")


def load_dataset() -> List[dict]:
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_progress() -> dict:
    if not os.path.exists(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def filter_dataset(
    data: List[dict],
    langs: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[dict]:
    result = data
    if langs:
        result = [d for d in result if d.get("language") in langs]
    if categories:
        result = [d for d in result if d.get("category") in categories]
    if start_date:
        s_dt = datetime.fromisoformat(start_date)
        result = [d for d in result if "created_at" in d and datetime.fromisoformat(d["created_at"]) >= s_dt]
    if end_date:
        e_dt = datetime.fromisoformat(end_date)
        result = [d for d in result if "created_at" in d and datetime.fromisoformat(d["created_at"]) <= e_dt]
    return result


def enrich_record(record: dict) -> dict:
    """Return record with cleaned text and extracted entities."""
    text = clean_text(record.get("content", ""))
    record = {**record, "clean_content": text, "entities": extract_entities(text)}
    return record

class ScrapeParams(BaseModel):
    lang: Optional[List[str]] | Optional[str] = None
    category: Optional[List[str]] | Optional[str] = None
    format: str = "all"
    plugin: str = "wikipedia"  # e.g. "infobox_parser" or "table_parser"

@app.post("/scrape")
async def scrape(params: ScrapeParams):
    langs = params.lang
    if isinstance(langs, str):
        langs = [langs]
    cats = params.category
    if isinstance(cats, str):
        cats = [cats]
    if cats is not None:
        cats = [sw.normalize_category(c) or c for c in cats]

    if params.plugin == "wikipedia":
        sw.main(langs, cats, params.format)
    else:
        from plugins import load_plugin, run_plugin

        plg = load_plugin(params.plugin)
        run_plugin(
            plg,
            langs or sw.Config.LANGUAGES,
            cats or list(sw.Config.CATEGORIES),
            params.format,
        )
    return {"status": "ok"}


@app.get("/records")
async def get_records(
    lang: Optional[List[str]] | Optional[str] = Query(None),
    category: Optional[List[str]] | Optional[str] = Query(None),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    langs = lang if isinstance(lang, list) else ([lang] if lang else None)
    cats = category if isinstance(category, list) else ([category] if category else None)
    data = load_dataset()
    filtered = filter_dataset(data, langs, cats, start_date, end_date)
    processed = await asyncio.gather(
        *(asyncio.to_thread(enrich_record, rec) for rec in filtered)
    )
    return processed


@app.get("/stats")
async def get_stats():
    return load_progress()


class RecordType(graphene.ObjectType):
    id = graphene.String()
    title = graphene.String()
    language = graphene.String()
    category = graphene.String()
    topic = graphene.String()
    subtopic = graphene.String()
    summary = graphene.String()
    created_at = graphene.String()


class Query(graphene.ObjectType):
    records = graphene.List(
        RecordType,
        lang=graphene.List(graphene.String),
        category=graphene.List(graphene.String),
        start_date=graphene.String(),
        end_date=graphene.String(),
    )

    def resolve_records(root, info, lang=None, category=None, start_date=None, end_date=None):
        data = load_dataset()
        filtered = filter_dataset(data, lang, category, start_date, end_date)
        return [enrich_record(rec) for rec in filtered]


schema = graphene.Schema(query=Query)


@app.post("/graphql")
async def graphql_endpoint(request: Request):
    body = await request.json()
    result = schema.execute(body.get("query"), variable_values=body.get("variables"))
    return JSONResponse(result.data)


@app.get("/health")
async def health_check():
    """Basic health check endpoint used by containers and load balancers."""
    return {"status": "ok"}
