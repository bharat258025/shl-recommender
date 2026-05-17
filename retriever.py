"""
retriever.py
Semantic search using chromadb's built-in ONNX embeddings.
Memory usage: ~80MB vs ~800MB with torch/sentence-transformers.
Fits comfortably within Render free tier (512MB limit).
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db")
CATALOG_PATH = os.getenv("CATALOG_PATH", "./data/catalog.json")


# ── Domain rules for reranking ────────────────────────────────────────────────

DOMAIN_RULES = [
    {
        "triggers": ["data analyst", "sql", "dashboard", "reporting", "bi ", "business intelligence", "tableau", "power bi"],
        "boost": ["SQL (New)", "Verify Numerical Reasoning", "Verify Verbal Reasoning"],
        "suppress": ["Management & Leadership Report (MLR)", "Sales Solution (SSCE)", "Contact Center Solution"],
    },
    {
        "triggers": ["java"],
        "boost": ["Java 8 (New)", "Verify Inductive Reasoning", "Technology Professional (TP1)"],
        "suppress": ["Sales Solution (SSCE)", "Contact Center Solution", "Dependability & Safety Instrument (DSI)"],
    },
    {
        "triggers": ["python", "data science", "machine learning", "ml engineer"],
        "boost": ["Python (New)", "Verify Inductive Reasoning", "Verify Numerical Reasoning"],
        "suppress": ["Sales Solution (SSCE)", "Contact Center Solution"],
    },
    {
        "triggers": ["javascript", "frontend", "react", "nodejs", "full-stack", "web developer"],
        "boost": ["JavaScript (New)", "Verify Inductive Reasoning"],
        "suppress": ["Sales Solution (SSCE)", "Contact Center Solution"],
    },
    {
        "triggers": ["sales manager", "head of sales", "vp sales", "director of sales"],
        "boost": ["Sales Solution (SSCE)", "OPQ32r", "Management & Leadership Report (MLR)"],
        "suppress": ["Universal Competency Report (UCR)", "Contact Center Solution"],
    },
    {
        "triggers": ["sales", "b2b", "b2c", "account manager", "business development", "revenue"],
        "boost": ["Sales Solution (SSCE)", "OPQ32r"],
        "suppress": ["Contact Center Solution", "Dependability & Safety Instrument (DSI)", "SQL (New)"],
    },
    {
        "triggers": ["manager", "director", "executive", "leadership", "head of", "c-suite"],
        "boost": ["OPQ32r", "Management & Leadership Report (MLR)"],
        "suppress": ["Contact Center Solution", "Dependability & Safety Instrument (DSI)"],
    },
    {
        "triggers": ["graduate", "campus", "fresh graduate", "entry level", "early career", "university", "college", "intern"],
        "boost": ["Graduate 8.0 (Short)", "Graduate Personality Questionnaire"],
        "suppress": ["Management & Leadership Report (MLR)", "Sales Solution (SSCE)", "Contact Center Solution"],
    },
    {
        "triggers": ["warehouse", "logistics", "operations", "safety", "manufacturing", "supply chain", "blue collar"],
        "boost": ["Dependability & Safety Instrument (DSI)", "Operational Assessment (OP5)"],
        "suppress": ["OPQ32r", "Management & Leadership Report (MLR)", "Sales Solution (SSCE)", "SQL (New)", "Java 8 (New)"],
    },
    {
        "triggers": ["contact center", "call center", "bpo", "inbound", "outbound", "customer support"],
        "boost": ["Contact Center Solution", "CustomerFirst (CF3)", "Workplace English (WE1)"],
        "suppress": ["Management & Leadership Report (MLR)", "SQL (New)", "Java 8 (New)"],
    },
    {
        "triggers": ["stakeholder", "client facing", "cross-functional", "collaborate", "teamwork"],
        "boost": ["OPQ32r", "Technology Professional (TP1)"],
        "suppress": [],
    },
    {
        "triggers": ["personality", "behavior", "culture fit", "soft skills"],
        "boost": ["OPQ32r", "Motivation Questionnaire (MQM5)"],
        "suppress": [],
    },
]


def _rerank(query: str, results: list[dict]) -> list[dict]:
    q = query.lower()
    scores = {r["name"]: 0.0 for r in results}
    name_to_result = {r["name"]: r for r in results}

    for rule in DOMAIN_RULES:
        if any(trigger in q for trigger in rule["triggers"]):
            for name in rule.get("boost", []):
                scores[name] = scores.get(name, 0.0) + 10.0
            for name in rule.get("suppress", []):
                scores[name] = scores.get(name, 0.0) - 5.0

    original_rank = {r["name"]: i for i, r in enumerate(results)}

    def sort_key(name):
        return (-scores.get(name, 0), original_rank.get(name, 999))

    all_names = sorted(scores.keys(), key=sort_key)
    final = []
    for name in all_names:
        if scores.get(name, 0) < 0 and len(final) >= 3:
            continue
        if name in name_to_result:
            final.append(name_to_result[name])

    return final[:10]


class SHLRetriever:
    def __init__(self):
        self._collection = None
        self._catalog: list[dict] = []
        self._catalog_by_name: dict[str, dict] = {}
        self._ef = None

    def load(self):
        if self._collection is not None:
            return

        if not Path(CHROMA_PERSIST_DIR).exists():
            raise RuntimeError(
                f"ChromaDB not found at {CHROMA_PERSIST_DIR}. "
                "Run `python build_vectorstore.py` first."
            )

        logger.info("Loading ONNX embedding function...")
        self._ef = embedding_functions.ONNXMiniLM_L6_V2()

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._collection = client.get_collection(
            name="shl_assessments",
            embedding_function=self._ef,
        )
        logger.info("ChromaDB loaded.")

        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            self._catalog = json.load(f)
        self._catalog_by_name = {item["name"].lower(): item for item in self._catalog}
        logger.info(f"Catalog loaded: {len(self._catalog)} assessments.")

    def _catalog_item_to_result(self, item: dict) -> dict:
        levels = ", ".join(item.get("job_levels", [])) if isinstance(item.get("job_levels"), list) else item.get("job_levels", "")
        keywords = ", ".join(item.get("keywords", [])) if isinstance(item.get("keywords"), list) else item.get("keywords", "")
        return {
            "name": item["name"],
            "url": item["url"],
            "test_type": item.get("test_type", ""),
            "description": item.get("description", ""),
            "job_levels": levels,
            "duration_minutes": str(item.get("duration_minutes", "")),
            "remote_testing": str(item.get("remote_testing", "")),
            "adaptive": str(item.get("adaptive", "")),
            "keywords": keywords,
        }

    def search(self, query: str, k: int = 10) -> list[dict]:
        if self._collection is None:
            raise RuntimeError("Retriever not loaded. Call .load() first.")

        # Semantic search
        results_raw = self._collection.query(
            query_texts=[query],
            n_results=min(15, self._collection.count()),
        )

        results = []
        if results_raw and results_raw["metadatas"]:
            for meta in results_raw["metadatas"][0]:
                results.append({
                    "name": meta.get("name", ""),
                    "url": meta.get("url", ""),
                    "test_type": meta.get("test_type", ""),
                    "description": meta.get("description", ""),
                    "job_levels": meta.get("job_levels", ""),
                    "duration_minutes": meta.get("duration_minutes", ""),
                    "remote_testing": meta.get("remote_testing", ""),
                    "adaptive": meta.get("adaptive", ""),
                    "keywords": meta.get("keywords", ""),
                })

        # Inject domain must-haves not in semantic top-15
        q = query.lower()
        existing_names = {r["name"] for r in results}
        for rule in DOMAIN_RULES:
            if any(trigger in q for trigger in rule["triggers"]):
                for name in rule.get("boost", []):
                    if name not in existing_names:
                        item = self._catalog_by_name.get(name.lower())
                        if item:
                            results.append(self._catalog_item_to_result(item))
                            existing_names.add(name)

        return _rerank(query, results)[:k]

    def get_by_name(self, name: str) -> Optional[dict]:
        return self._catalog_by_name.get(name.lower())

    def get_all(self) -> list[dict]:
        return self._catalog

    def format_for_prompt(self, results: list[dict]) -> str:
        if not results:
            return "No assessments found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. [{r['name']}] (Type:{r['test_type']}) — {r['description'][:120]}..."
                f"\n   URL: {r['url']}"
                f"\n   Levels: {r['job_levels']} | Duration: {r['duration_minutes']}min"
            )
        return "\n\n".join(lines)


retriever = SHLRetriever()