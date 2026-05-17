# SHL Assessment Recommender — Approach Document

**Role:** AI Intern — SHL Labs | **Stack:** FastAPI, LangChain, ChromaDB, sentence-transformers, Groq

---

## 1. Problem Decomposition

The core challenge: translate vague hiring intent into a grounded shortlist from a fixed catalog, through multi-turn dialogue, without hallucinating assessments that don't exist.

I decomposed this into four sub-problems:
1. **Catalog representation** — how to store and retrieve assessments by semantic meaning
2. **Intent routing** — deciding whether to clarify, recommend, refine, compare, or refuse
3. **Grounded generation** — giving the LLM only catalog-sourced context so it cannot invent URLs
4. **Robustness** — handling vague input, mid-conversation edits, and out-of-scope queries

---

## 2. Architecture

```
POST /chat (full history)
        ↓
Intent Classifier (LLM, 1 call)
  → CLARIFY / RECOMMEND / REFINE / COMPARE / OUT_OF_SCOPE
        ↓
Hybrid Retriever
  Stage 1: ChromaDB MMR semantic search (top-15 candidates)
  Stage 2: Rule-based injection (force domain must-haves)
  Stage 3: Score-based reranking (boost/suppress by domain)
        ↓
LLM Response Generation (catalog context injected into prompt)
        ↓
Post-parse guardrails (URL whitelist, rec cap)
        ↓
ChatResponse {reply, recommendations, end_of_conversation}
```

**Stateless design:** full conversation history sent on every call. No server-side session state.

---

## 3. Key Design Decisions

### Two-Pass LLM Calls
A cheap classifier call first determines intent, then a second call generates the response with the appropriate action instruction. This prevents the agent from recommending on vague queries and makes each step independently debuggable.

**Trade-off:** Two LLM calls per turn (~2x latency). Mitigated by using Groq (LPU inference, ~0.5s per call).

### Hybrid Retrieval (Most Important Decision)
Pure semantic search gave Recall@10 of ~0.43 — the LLM would retrieve MLR/OPQ32r for data analyst queries because personality tests are semantically close to "professional roles." Pure LLM prompting also failed because free-tier models inconsistently follow detailed instructions.

**Solution:** Three-stage hybrid retrieval:
- **Semantic MMR search** fetches 15 candidates (Maximal Marginal Relevance for diversity)
- **Rule injection** force-adds domain must-haves not in semantic top-15 (e.g., SQL(New) for data analyst queries)
- **Score reranking** applies +10 boost / -5 suppression per domain rule match

This lifted Mean Recall@10 from 0.43 → 0.88 without depending on LLM instruction-following.

### Local Embeddings
`sentence-transformers/all-MiniLM-L6-v2` runs on CPU — no API cost, deterministic, fast enough (~200ms). Catalog entries embedded with rich text: name + type + description + job levels + keywords. Keyword enrichment alone improved recall from 0.43 to 0.68 before the rule reranker was added.

### Context Engineering
Retrieved catalog entries injected into system prompt at query time. The LLM only sees real catalog entries — it cannot hallucinate a URL it was never shown. Post-parse URL whitelist (`https://www.shl.com/`) provides a hard second guardrail.

Action instruction embedded into the final HumanMessage (not as a trailing SystemMessage) for cross-provider compatibility with Groq, Gemini, and OpenRouter.

---

## 4. Guardrails

| Guardrail | Implementation |
|---|---|
| URL whitelist | Every recommendation URL validated post-parse; non-SHL URLs silently dropped |
| Hallucination prevention | LLM only given real catalog entries in context |
| Scope enforcement | Intent classifier rejects off-topic before any retrieval |
| Rec cap | Hard-capped at 10 in code, not just in prompt |
| JSON robustness | Regex fence stripping + boundary detection before `json.loads` |
| Rate limit recovery | Exponential backoff with 4 retries |

---

## 5. Evaluation

**Hard evals (11/11 passed):**
- Schema compliance on every response
- No recommendations on vague first message
- Recommendations only from catalog (URL whitelist)
- Refine updates shortlist without restarting
- Compare uses catalog data only
- Out-of-scope and prompt injection refused

**Recall@10 results:**

| Trace | Recall@10 |
|---|---|
| Java developer mid-level | 0.75 |
| Sales manager B2B | 1.00 |
| Graduate campus hiring | 0.67 |
| Data analyst SQL | 1.00 |
| Warehouse logistics safety | 1.00 |
| **Mean Recall@10** | **0.883** |

---

## 6. What Didn't Work

| Attempt | Problem | Fix |
|---|---|---|
| Pure semantic search | Recalled wrong domain tests (MLR for data analyst) | Added rule-based reranker |
| Single large prompt | Agent recommended on vague queries | Split into classifier + action prompts |
| Embedding description only | Low recall for role-based queries | Added keywords + job levels to embedding text |
| Trailing SystemMessage | Broke Gemini (requires last msg = user) | Embedded instruction into final HumanMessage |
| `openrouter/auto` | Hit 200 req/day free limit | Switched to Groq (14,400 req/day, faster) |
| Detailed prompt selection rules | Inconsistent compliance across free models | Moved selection logic into retrieval layer (deterministic) |

---

## 7. Stack Justification

| Component | Choice | Reason |
|---|---|---|
| Web framework | FastAPI | Native async, Pydantic validation, auto-docs for evaluator testing |
| LLM | Groq `llama-3.3-70b-versatile` | 14,400 free req/day, ~0.5s latency, reliable JSON output |
| Embeddings | `all-MiniLM-L6-v2` (local) | No API cost, deterministic, 384-dim, fast on CPU |
| Vector store | ChromaDB | Zero-config, persists to disk, good LangChain integration |
| Retrieval | Hybrid MMR + rules | Deterministic recall that doesn't depend on LLM compliance |
| Deployment | Render (Docker) | Free tier, HTTPS, Docker support, env var secrets |

**AI tools used:** Claude assisted with initial scaffolding. All architecture decisions, retrieval design, prompt engineering, and debugging were done independently. Code reviewed line-by-line — every choice can be defended.

---

*~700 lines of Python. 11/11 behavior probes passing. Mean Recall@10: 0.883.*