import json
import os
from pathlib import Path

import streamlit as st
import psutil
import requests

PROGRESS_FILE = Path("logs/progress.json")
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
PROM_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")


def load_progress():
    try:
        resp = requests.get(f"{API_BASE}/stats", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        if PROGRESS_FILE.exists():
            try:
                with PROGRESS_FILE.open() as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def load_metrics():
    metrics = {
        "success": 0,
        "error": 0,
        "block": 0,
        "pages": 0,
        "failures": 0,
    }
    try:
        for name, key in {
            "success": "scrape_success_total",
            "error": "scrape_error_total",
            "block": "scrape_block_total",
            "pages": "pages_scraped_total",
            "failures": "requests_failed_total",
        }.items():
            resp = requests.get(
                f"{PROM_URL}/api/v1/query",
                params={"query": key},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("data", {}).get("result"):
                metrics[name] = float(data["data"]["result"][0]["value"][1])
    except Exception:
        pass
    return metrics


def main():
    st.title("Scraper Progress Dashboard")

    progress = load_progress()
    metrics = load_metrics()
    pages_processed = progress.get("pages_processed", 0)
    clusters = progress.get("clusters", [])
    topics = progress.get("topics", [])
    languages = progress.get("languages", [])

    st.metric("Pages processed", pages_processed)
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    st.metric("CPU usage", f"{cpu}%")
    st.metric("RAM usage", f"{ram}%")

    st.metric("Scrape success", metrics["success"])
    st.metric("Scrape errors", metrics["error"])
    st.metric("Scrape blocks", metrics["block"])

    st.subheader("Clusters")
    st.write(clusters)

    st.subheader("Topics")
    st.write(topics)

    st.subheader("Languages")
    st.write(languages)


if __name__ == "__main__":
    main()
