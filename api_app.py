from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import scraper_wiki as sw

app = FastAPI()

class ScrapeParams(BaseModel):
    lang: Optional[List[str]] | Optional[str] = None
    category: Optional[List[str]] | Optional[str] = None
    format: str = "all"
    plugin: str = "wikipedia"

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
