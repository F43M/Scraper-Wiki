import json
import os
from pathlib import Path

import streamlit as st
import psutil
import requests

PROGRESS_FILE = Path("logs/progress.json")
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")


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


def main():
    st.title("Scraper Progress Dashboard")

    progress = load_progress()
    pages_processed = progress.get("pages_processed", 0)
    clusters = progress.get("clusters", [])
    topics = progress.get("topics", [])
    languages = progress.get("languages", [])

    st.metric("Pages processed", pages_processed)
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    st.metric("CPU usage", f"{cpu}%")
    st.metric("RAM usage", f"{ram}%")

    st.subheader("Clusters")
    st.write(clusters)

    st.subheader("Topics")
    st.write(topics)

    st.subheader("Languages")
    st.write(languages)


if __name__ == "__main__":
    main()
