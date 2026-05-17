"""
main.py
FastAPI service for the SHL Assessment Recommender.

Endpoints:
  GET  /health  — readiness check
  POST /chat    — stateless conversational agent

The service is stateless: every /chat call carries the full conversation history.
State lives entirely in the client (or the evaluator harness).
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from agent import run_agent
from retriever import retriever

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Pydantic models ───────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=20)

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if v[0].role != "user":
            raise ValueError("First message must be from 'user'")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = []
    end_of_conversation: bool = False


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load retriever (embeddings + ChromaDB) at startup."""
    logger.info("Starting SHL Recommender service...")
    start = time.time()
    try:
        retriever.load()
        elapsed = time.time() - start
        logger.info(f"Retriever loaded in {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load retriever: {e}")
        raise
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL Individual Test Solutions.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins (for evaluator harness)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request logging middleware ────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(f"{request.method} {request.url.path} → {response.status_code} [{elapsed:.3f}s]")
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Readiness check. Returns 200 OK when the service is ready.
    The evaluator allows up to 2 minutes for cold start.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless conversational endpoint.
    Accepts full conversation history; returns next agent reply + optional shortlist.
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Enforce turn cap (max 8 turns = 4 user + 4 assistant, evaluator enforces this)
    if len(messages) > 16:
        logger.warning("Message list exceeds expected size; truncating to last 16.")
        messages = messages[-16:]

    try:
        result = run_agent(messages)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="The agent encountered an error. Please try again.",
        )

    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r["test_type"],
            )
            for r in result.get("recommendations", [])
        ],
        end_of_conversation=result.get("end_of_conversation", False),
    )


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=False,  # Disable reload in production
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
