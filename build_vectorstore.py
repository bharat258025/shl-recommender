"""
build_vectorstore.py
Builds and persists the ChromaDB vector store from catalog.json.
Run this ONCE before starting the server: python build_vectorstore.py
"""

import json
import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CATALOG_PATH = os.getenv("CATALOG_PATH", "./data/catalog.json")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def load_catalog(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_document(item: dict) -> Document:
    test_type_labels = {
        "A": "Ability / Aptitude",
        "P": "Personality / Behavior",
        "K": "Knowledge / Skills",
        "S": "Simulation / Practical",
        "B": "Behavioral / Situational Judgement",
    }
    test_type_label = test_type_labels.get(item.get("test_type", ""), "Unknown")
    levels = ", ".join(item.get("job_levels", []))
    keywords = ", ".join(item.get("keywords", []))

    content = f"""
Assessment Name: {item['name']}
Test Type: {test_type_label} ({item.get('test_type', '')})
Job Levels: {levels}
Duration: {item.get('duration_minutes', 'N/A')} minutes
Remote Testing: {'Yes' if item.get('remote_testing') else 'No'}
Adaptive/IRT: {'Yes' if item.get('adaptive') else 'No'}
Description: {item['description']}
Keywords: {keywords}
    """.strip()

    metadata = {
        "name": item["name"],
        "url": item["url"],
        "test_type": item.get("test_type", ""),
        "job_levels": levels,
        "duration_minutes": str(item.get("duration_minutes", "")),
        "remote_testing": str(item.get("remote_testing", "")),
        "adaptive": str(item.get("adaptive", "")),
        "keywords": keywords,
        "description": item["description"],
    }

    return Document(page_content=content, metadata=metadata)


def build_vectorstore():
    logger.info(f"Loading catalog from {CATALOG_PATH}")
    catalog = load_catalog(CATALOG_PATH)
    logger.info(f"Loaded {len(catalog)} assessments")

    documents = [build_document(item) for item in catalog]
    logger.info("Documents built, loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info(f"Embedding model loaded: {EMBEDDING_MODEL}")

    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    logger.info("Building ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="shl_assessments",
    )
    logger.info(f"Vector store built with {len(documents)} documents -> {CHROMA_PERSIST_DIR}")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()
    print("\n✅ Vector store built successfully. You can now run the server.")