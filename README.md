# SHL Assessment Recommender

A conversational agent that recommends SHL Individual Test Solutions based on hiring requirements. Built with FastAPI, LangChain, ChromaDB, and sentence-transformers.

**Test Results: 11/11 behavior tests passed | Mean Recall@10: 0.883**

---

## Project Structure

```
shl-recommender/
├── main.py                  # FastAPI app — GET /health, POST /chat
├── agent.py                 # LLM agent: intent classification + response
├── retriever.py             # Hybrid semantic + rule-based retrieval
├── build_vectorstore.py     # One-time script: embeds catalog into ChromaDB
├── data/
│   └── catalog.json         # 30 SHL Individual Test Solutions
├── vectorstore/             # Auto-created by build_vectorstore.py
├── tests/
│   └── test_agent.py        # 11 behavior probes + Recall@10 evaluation
├── requirements.txt
├── .env.example
├── Dockerfile
├── render.yaml
├── APPROACH.md
└── README.md
```

---

## Local Setup

### Prerequisites
- Python 3.10 or 3.11
- Free Groq API key from https://console.groq.com (recommended — 14,400 req/day free)

### Step 1 — Clone and open
```bash
git clone https://github.com/bharat258025/shl-recommender
cd shl-recommender
```

### Step 2 — Create virtual environment
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Configure environment
```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_key_here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./vectorstore/chroma_db
CATALOG_PATH=./data/catalog.json
```

### Step 5 — Build vector store (run once)
```bash
python build_vectorstore.py
```

### Step 6 — Start server
```bash
python main.py
```

### Step 7 — Test
```bash
# Health check (browser works too)
curl http://localhost:8000/health

# Interactive API docs
open http://localhost:8000/docs

# Chat test
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I am hiring a mid-level Java developer who works with stakeholders"}]}'
```

### Step 8 — Run test suite
```bash
python tests/test_agent.py
```

---

## API Reference

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, 4 years experience"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are assessments for a mid-level Java developer...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

**Test type codes:** A=Ability, P=Personality, K=Knowledge, S=Simulation, B=Behavioral

---

## LLM Provider Options

| Provider | Free Tier | Speed | Setup |
|---|---|---|---|
| **Groq** (recommended) | 14,400 req/day | Very fast | console.groq.com |
| OpenRouter | 200 req/day | Medium | openrouter.ai |
| Gemini | 1,500 req/day | Medium | aistudio.google.com |
| OpenAI | Paid | Fast | platform.openai.com |

Switch provider by changing `.env`:
```env
# Groq
GROQ_API_KEY=gsk_...
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile

# OpenRouter
OPENROUTER_API_KEY=sk-or-v1-...
LLM_PROVIDER=openrouter
LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free
```