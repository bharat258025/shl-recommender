"""
retriever.py
Semantic search over SHL catalog + rule-based reranking.

Two-stage retrieval:
  1. ChromaDB MMR search (semantic similarity, broad recall)
  2. Rule-based reranker (boosts domain-specific must-haves, kills wrong-domain results)

This hybrid approach means recall doesn't depend on the LLM following prompt instructions.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CATALOG_PATH = os.getenv("CATALOG_PATH", "./data/catalog.json")


# ── Domain rules ──────────────────────────────────────────────────────────────
# Each rule: if ANY trigger keyword found in query → boost these assessments
# Scores are additive. Top-scored results rise to the top.

DOMAIN_RULES = [
    # Data analyst / SQL / BI
    {
        "triggers": ["data analyst", "sql", "dashboard", "reporting", "bi ", "business intelligence", "tableau", "power bi"],
        "boost": ["SQL (New)", "Verify Numerical Reasoning", "Verify Verbal Reasoning"],
        "suppress": ["Management & Leadership Report (MLR)", "Sales Solution (SSCE)", "Contact Center Solution"],
    },
    # Java developer
    {
        "triggers": ["java"],
        "boost": ["Java 8 (New)", "Verify Inductive Reasoning", "Technology Professional (TP1)"],
        "suppress": ["Sales Solution (SSCE)", "Contact Center Solution", "Dependability & Safety Instrument (DSI)"],
    },
    # Python / data science / ML
    {
        "triggers": ["python", "data science", "machine learning", "ml engineer"],
        "boost": ["Python (New)", "Verify Inductive Reasoning", "Verify Numerical Reasoning"],
        "suppress": ["Sales Solution (SSCE)", "Contact Center Solution"],
    },
    # JavaScript / frontend / web
    {
        "triggers": ["javascript", "frontend", "react", "nodejs", "full-stack", "fullstack", "web developer"],
        "boost": ["JavaScript (New)", "Verify Inductive Reasoning"],
        "suppress": ["Sales Solution (SSCE)", "Contact Center Solution"],
    },
    # Sales
    {
        "triggers": ["sales", "b2b", "b2c", "account manager", "business development", "revenue", "quota"],
        "boost": ["Sales Solution (SSCE)", "OPQ32r"],
        "suppress": ["Contact Center Solution", "Dependability & Safety Instrument (DSI)", "SQL (New)"],
    },
    # Sales MANAGER specifically — add MLR
    {
        "triggers": ["sales manager", "head of sales", "vp sales", "director of sales"],
        "boost": ["Sales Solution (SSCE)", "OPQ32r", "Management & Leadership Report (MLR)"],
        "suppress": ["Universal Competency Report (UCR)", "Contact Center Solution"],
    },
    # Manager / leadership (non-sales)
    {
        "triggers": ["manager", "director", "executive", "leadership", "vp ", "head of", "c-suite", "cto", "ceo"],
        "boost": ["OPQ32r", "Management & Leadership Report (MLR)"],
        "suppress": ["Contact Center Solution", "Dependability & Safety Instrument (DSI)"],
    },
    # Graduate / campus / fresh
    {
        "triggers": ["graduate", "campus", "fresh graduate", "entry level", "early career", "university", "college", "intern"],
        "boost": ["Graduate 8.0 (Short)", "Graduate Personality Questionnaire"],
        "suppress": ["Management & Leadership Report (MLR)", "Sales Solution (SSCE)", "Contact Center Solution"],
    },
    # Warehouse / logistics / operations / safety
    {
        "triggers": ["warehouse", "logistics", "operations", "safety", "forklift", "manufacturing", "supply chain", "blue collar"],
        "boost": ["Dependability & Safety Instrument (DSI)", "Operational Assessment (OP5)"],
        "suppress": ["OPQ32r", "Management & Leadership Report (MLR)", "Sales Solution (SSCE)", "SQL (New)", "Java 8 (New)"],
    },
    # Contact center / BPO / call center
    {
        "triggers": ["contact center", "call center", "bpo", "inbound", "outbound", "customer support", "helpdesk"],
        "boost": ["Contact Center Solution", "CustomerFirst (CF3)", "Workplace English (WE1)"],
        "suppress": ["Management & Leadership Report (MLR)", "SQL (New)", "Java 8 (New)"],
    },
    # Stakeholder work → add personality/SJT
    {
        "triggers": ["stakeholder", "client facing", "cross-functional", "collaborate", "teamwork"],
        "boost": ["OPQ32r", "Technology Professional (TP1)"],
        "suppress": [],
    },
    # Personality explicitly requested
    {
        "triggers": ["personality", "behavior", "culture fit", "soft skills", "work style"],
        "boost": ["OPQ32r", "Motivation Questionnaire (MQM5)"],
        "suppress": [],
    },
]


def _rerank(query: str, results: list[dict]) -> list[dict]:
    """
    Apply domain rules to rerank retrieved results.
    - Boosted assessments get +10 score each time a rule matches
    - Suppressed assessments get -5 score each time a rule matches
    - Results are sorted by final score descending
    - Suppressed-only results (score < 0) are removed if we have enough boosted ones
    """
    q = query.lower()
    scores = {r["name"]: 0.0 for r in results}

    # Build a name→result map for fast lookup
    name_to_result = {r["name"]: r for r in results}

    # Also build full catalog map for boosted items not in top-10
    all_names = {r["name"] for r in results}

    matched_boosts = set()  # track which names were boosted

    for rule in DOMAIN_RULES:
        if any(trigger in q for trigger in rule["triggers"]):
            for name in rule.get("boost", []):
                if name not in scores:
                    scores[name] = 0.0
                scores[name] += 10.0
                matched_boosts.add(name)
            for name in rule.get("suppress", []):
                if name not in scores:
                    scores[name] = 0.0
                scores[name] -= 5.0

    # Sort by score descending, then original rank for ties
    original_rank = {r["name"]: i for i, r in enumerate(results)}

    def sort_key(name):
        return (-scores.get(name, 0), original_rank.get(name, 999))

    all_names_ordered = sorted(scores.keys(), key=sort_key)

    # Rebuild result list — only keep items that are in original results OR were boosted
    final = []
    for name in all_names_ordered:
        if scores.get(name, 0) < 0 and len(final) >= 3:
            continue  # Skip suppressed items once we have enough
        if name in name_to_result:
            final.append(name_to_result[name])
        # Boosted items not in original results are added from catalog later (in SHLRetriever.search)

    return final[:10]


class SHLRetriever:
    def __init__(self):
        self._vectorstore: Optional[Chroma] = None
        self._catalog: list[dict] = []
        self._embeddings = None
        self._catalog_by_name: dict[str, dict] = {}

    def load(self):
        if self._vectorstore is not None:
            return

        logger.info("Loading HuggingFace embedding model...")
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if not Path(CHROMA_PERSIST_DIR).exists():
            raise RuntimeError(
                f"ChromaDB not found at {CHROMA_PERSIST_DIR}. "
                "Run `python build_vectorstore.py` first."
            )

        self._vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self._embeddings,
            collection_name="shl_assessments",
        )
        logger.info("ChromaDB loaded.")

        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            self._catalog = json.load(f)
        self._catalog_by_name = {item["name"].lower(): item for item in self._catalog}
        logger.info(f"Catalog loaded: {len(self._catalog)} assessments.")

    def _catalog_item_to_result(self, item: dict) -> dict:
        levels = ", ".join(item.get("job_levels", [])) if isinstance(item.get("job_levels"), list) else item.get("job_levels", "")
        return {
            "name": item["name"],
            "url": item["url"],
            "test_type": item.get("test_type", ""),
            "description": item.get("description", ""),
            "job_levels": levels,
            "duration_minutes": str(item.get("duration_minutes", "")),
            "remote_testing": str(item.get("remote_testing", "")),
            "adaptive": str(item.get("adaptive", "")),
            "keywords": ", ".join(item.get("keywords", [])) if isinstance(item.get("keywords"), list) else item.get("keywords", ""),
        }

    def search(self, query: str, k: int = 10) -> list[dict]:
        """
        Two-stage retrieval:
        1. Semantic MMR search (fetch top-15 candidates)
        2. Rule-based reranking + forced injection of domain must-haves
        """
        if self._vectorstore is None:
            raise RuntimeError("Retriever not loaded. Call .load() first.")

        # Stage 1: semantic search — fetch more candidates than needed
        docs = self._vectorstore.max_marginal_relevance_search(
            query, k=15, fetch_k=30
        )
        results = []
        for doc in docs:
            meta = doc.metadata
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

        # Stage 2: inject boosted items that didn't make the semantic top-15
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

        # Stage 3: rerank
        reranked = _rerank(query, results)
        return reranked[:k]

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