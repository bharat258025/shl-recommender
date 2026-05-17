"""
build_vectorstore.py
Builds ChromaDB vector store using chromadb's built-in embedding function.
Uses ONNXRuntime instead of torch — works within 512MB RAM on free hosting.
Run once: python build_vectorstore.py
"""

import json
import os
import logging
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CATALOG_PATH = os.getenv("CATALOG_PATH", "./data/catalog.json")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def load_catalog(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_document_text(item: dict) -> str:
    """Rich text for embedding — more context = better retrieval."""
    test_type_labels = {
        "A": "Ability / Aptitude",
        "P": "Personality / Behavior",
        "K": "Knowledge / Skills",
        "S": "Simulation / Practical",
        "B": "Behavioral / Situational Judgement",
    }
    test_type_label = test_type_labels.get(item.get("test_type", ""), "Unknown")
    levels = ", ".join(item.get("job_levels", [])) if isinstance(item.get("job_levels"), list) else item.get("job_levels", "")
    keywords = ", ".join(item.get("keywords", [])) if isinstance(item.get("keywords"), list) else item.get("keywords", "")

    return f"""Assessment Name: {item['name']}
Test Type: {test_type_label} ({item.get('test_type', '')})
Job Levels: {levels}
Duration: {item.get('duration_minutes', 'N/A')} minutes
Remote Testing: {'Yes' if item.get('remote_testing') else 'No'}
Description: {item['description']}
Keywords: {keywords}""".strip()


def build_vectorstore():
    logger.info(f"Loading catalog from {CATALOG_PATH}")
    catalog = load_catalog(CATALOG_PATH)
    logger.info(f"Loaded {len(catalog)} assessments")

    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    # Use chromadb's built-in ONNX embedding function — no torch needed
    logger.info(f"Setting up ONNX embedding function: {EMBEDDING_MODEL}")
    ef = embedding_functions.ONNXMiniLM_L6_V2()

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection("shl_assessments")
        logger.info("Deleted existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name="shl_assessments",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    documents = []
    metadatas = []
    ids = []

    for i, item in enumerate(catalog):
        doc_text = build_document_text(item)
        levels = ", ".join(item.get("job_levels", [])) if isinstance(item.get("job_levels"), list) else item.get("job_levels", "")
        keywords = ", ".join(item.get("keywords", [])) if isinstance(item.get("keywords"), list) else item.get("keywords", "")

        documents.append(doc_text)
        metadatas.append({
            "name": item["name"],
            "url": item["url"],
            "test_type": item.get("test_type", ""),
            "job_levels": levels,
            "duration_minutes": str(item.get("duration_minutes", "")),
            "remote_testing": str(item.get("remote_testing", "")),
            "adaptive": str(item.get("adaptive", "")),
            "keywords": keywords,
            "description": item["description"],
        })
        ids.append(f"assessment_{i}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    logger.info(f"Vector store built with {len(documents)} documents → {CHROMA_PERSIST_DIR}")
    return collection


if __name__ == "__main__":
    build_vectorstore()
    print("\n✅ Vector store built successfully. You can now run the server.")