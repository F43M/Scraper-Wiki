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
        "avg_time": 0,
        "retries": 0,
        "avg_session": 0,
    }
    try:
        for name, key in {
            "success": "scrape_success_total",
            "error": "scrape_error_total",
            "block": "scrape_block_total",
            "pages": "pages_scraped_total",
            "failures": "requests_failed_total",
            "retries": "request_retries_total",
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

        # Calculate average processing time
        resp = requests.get(
            f"{PROM_URL}/api/v1/query",
            params={"query": "page_processing_seconds_sum"},
            timeout=5,
        )
        resp.raise_for_status()
        sum_data = resp.json()
        resp = requests.get(
            f"{PROM_URL}/api/v1/query",
            params={"query": "page_processing_seconds_count"},
            timeout=5,
        )
        resp.raise_for_status()
        count_data = resp.json()
        if sum_data.get("data", {}).get("result") and count_data.get("data", {}).get("result"):
            total = float(sum_data["data"]["result"][0]["value"][1])
            count = float(count_data["data"]["result"][0]["value"][1])
            if count:
                metrics["avg_time"] = total / count

        # Average session duration
        resp = requests.get(
            f"{PROM_URL}/api/v1/query",
            params={"query": "scrape_session_seconds_sum"},
            timeout=5,
        )
        resp.raise_for_status()
        sess_sum = resp.json()
        resp = requests.get(
            f"{PROM_URL}/api/v1/query",
            params={"query": "scrape_session_seconds_count"},
            timeout=5,
        )
        resp.raise_for_status()
        sess_count = resp.json()
        if sess_sum.get("data", {}).get("result") and sess_count.get("data", {}).get("result"):
            total = float(sess_sum["data"]["result"][0]["value"][1])
            count = float(sess_count["data"]["result"][0]["value"][1])
            if count:
                metrics["avg_session"] = total / count
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
    st.metric("Avg processing time (s)", round(metrics["avg_time"], 2))
    st.metric("Avg session time (s)", round(metrics["avg_session"], 2))

    st.metric("Scrape success", metrics["success"])
    st.metric("Scrape errors", metrics["error"])
    st.metric("Scrape blocks", metrics["block"])
    st.metric("Request retries", metrics["retries"])

    st.subheader("Clusters")
    st.write(clusters)

    st.subheader("Topics")
    st.write(topics)

    st.subheader("Languages")
    st.write(languages)


if __name__ == "__main__":
    main()
