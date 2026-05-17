"""
agent.py - SHL Assessment Recommender Agent
Supports OpenRouter (recommended), OpenAI, Groq, and Gemini.
"""

import json
import logging
import os
import re
import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from retriever import retriever

load_dotenv()
logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower().strip()
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free").strip()


def _get_llm():
    if LLM_PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in your .env file")
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=1024,
            timeout=25,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/shl-recommender",
                "X-Title": "SHL Assessment Recommender",
            },
        )
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in your .env file")
        return ChatOpenAI(model=LLM_MODEL, temperature=0.1, max_tokens=1024, timeout=25, api_key=api_key)
    elif LLM_PROVIDER == "groq":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in your .env file")
        return ChatOpenAI(model=LLM_MODEL, temperature=0.1, max_tokens=1024, timeout=25, api_key=api_key, base_url="https://api.groq.com/openai/v1")
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set in your .env file")
        return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1, max_output_tokens=1024, timeout=25, google_api_key=api_key)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'")


# ── Prompts ───────────────────────────────────────────────────────────────────

INTENT_SYSTEM = """You classify hiring manager requests for SHL assessments.

Output EXACTLY one word from: CLARIFY, RECOMMEND, REFINE, COMPARE, OUT_OF_SCOPE

RECOMMEND when the message contains ANY of:
- A job title (e.g. "Java developer", "sales manager", "data analyst", "graduate", "fresh graduate")
- A seniority level (e.g. "mid-level", "senior", "entry level", "campus hiring")
- A job description or responsibilities
- A domain (e.g. "contact center", "warehouse", "logistics", "consulting")

CLARIFY only when the message is completely generic with NO role/domain info:
- "I need an assessment" → CLARIFY
- "Help me" → CLARIFY
- "What assessments do you have?" → CLARIFY

REFINE when the user changes or adds to a previous shortlist.
COMPARE when the user asks to compare two named SHL assessments.
OUT_OF_SCOPE for legal questions, salary questions, or non-SHL topics.

Examples:
"Hiring a Java developer" → RECOMMEND
"Campus hiring for fresh graduates" → RECOMMEND
"Data analyst with SQL skills" → RECOMMEND
"Sales manager B2B" → RECOMMEND
"Warehouse staff, safety important" → RECOMMEND
"Add personality test" → REFINE
"Difference between OPQ and MQ?" → COMPARE
"Is it legal to test candidates?" → OUT_OF_SCOPE
"I need an assessment" → CLARIFY

Output ONE word only."""

MAIN_SYSTEM = """You are an SHL Assessment Recommender helping hiring managers choose the right SHL assessments.

SCOPE: ONLY discuss SHL assessments. Refuse all other topics politely.

CATALOG CONTEXT (use ONLY these — never invent assessments or URLs):
{catalog_context}

RULES:
1. CLARIFY: Ask ONE focused question if the user has given no role or domain info.
2. RECOMMEND: Provide 1-10 assessments. Pick the most relevant from the catalog. Include exact name and URL.
3. REFINE: Update the shortlist when constraints change. Acknowledge what changed.
4. COMPARE: Use ONLY catalog data — never use outside knowledge.
5. REFUSE: Politely decline non-SHL topics.

SELECTION GUIDANCE — follow these precisely:

TECHNICAL / DEVELOPER roles (Java, Python, JS, C++):
  MUST include: the relevant language test (e.g. Java 8 (New) for Java roles)
  MUST include: Verify Inductive Reasoning (analytical thinking for code)
  ADD if stakeholder work mentioned: Technology Professional (TP1) + OPQ32r
  ADD if data/numbers involved: Verify Numerical Reasoning

DATA ANALYST / BI / REPORTING roles:
  MUST include: SQL (New) — primary technical test
  MUST include: Verify Numerical Reasoning — core for data work
  MUST include: Verify Verbal Reasoning — for presenting insights
  DO NOT add personality/leadership tests unless explicitly asked

SALES roles:
  MUST include: Sales Solution (SSCE)
  MUST include: OPQ32r (personality for sales)
  If MANAGER level: MUST also include Management & Leadership Report (MLR)
  DO NOT include Universal Competency Report (UCR) — use MLR for managers

MANAGER / LEADERSHIP / EXECUTIVE roles:
  MUST include: OPQ32r
  MUST include: Management & Leadership Report (MLR)

GRADUATE / CAMPUS / FRESH GRADUATE / ENTRY LEVEL roles:
  MUST include: Graduate 8.0 (Short)
  MUST include: Graduate Personality Questionnaire
  ADD: Verify Numerical Reasoning or Verify Verbal Reasoning as appropriate

WAREHOUSE / LOGISTICS / OPERATIONS / SAFETY roles:
  MUST include: Dependability & Safety Instrument (DSI)
  MUST include: Operational Assessment (OP5)

CONTACT CENTER / BPO / CALL CENTER roles:
  MUST include: Contact Center Solution
  ADD: CustomerFirst (CF3), Workplace English (WE1)

CRITICAL: Every URL must start with https://www.shl.com/ and come from the catalog above. Never fabricate URLs."""

RECOMMEND_INSTRUCTION = """Respond with ONLY valid JSON, no markdown, no code fences:
{
  "reply": "Brief explanation of why these assessments fit the role.",
  "recommendations": [
    {"name": "Assessment Name", "url": "https://www.shl.com/...", "test_type": "X"}
  ],
  "end_of_conversation": false
}
Include 1-10 relevant assessments. Use exact names and URLs from catalog."""

CLARIFY_INSTRUCTION = """Respond with ONLY valid JSON, no markdown, no code fences:
{
  "reply": "Your single clarifying question about the role or domain.",
  "recommendations": [],
  "end_of_conversation": false
}"""

COMPARE_INSTRUCTION = """Respond with ONLY valid JSON, no markdown, no code fences:
{
  "reply": "Your comparison using only the catalog context provided.",
  "recommendations": [],
  "end_of_conversation": false
}"""

OUT_OF_SCOPE_INSTRUCTION = """Respond with ONLY valid JSON, no markdown, no code fences:
{
  "reply": "I can only help with SHL assessment selection. What role are you hiring for?",
  "recommendations": [],
  "end_of_conversation": false
}"""

ACTION_MAP = {
    "CLARIFY": CLARIFY_INSTRUCTION,
    "RECOMMEND": RECOMMEND_INSTRUCTION,
    "REFINE": RECOMMEND_INSTRUCTION,
    "COMPARE": COMPARE_INSTRUCTION,
    "OUT_OF_SCOPE": OUT_OF_SCOPE_INSTRUCTION,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_search_query(messages: list[dict]) -> str:
    """
    Build a rich search query from conversation history.
    Concatenates all user messages (not just last 4) so refinements
    don't lose the original role context.
    """
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_msgs)  # Use ALL user messages for full context


def _safe_json_parse(text: str) -> dict:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found. Raw: {text[:300]}")
    return json.loads(text[start:end])


def _sanitize_recommendations(recs: list[dict]) -> list[dict]:
    sanitized = []
    for r in recs:
        url = r.get("url", "")
        if url.startswith("https://www.shl.com/"):
            sanitized.append({
                "name": r.get("name", ""),
                "url": url,
                "test_type": r.get("test_type", ""),
            })
        else:
            logger.warning(f"Dropped invalid URL: {url}")
    return sanitized[:10]


def _invoke_with_retry(llm, messages, max_retries: int = 4) -> str:
    """
    Retry with exponential backoff for rate limit errors.
    Waits: 5s, 15s, 30s, 60s — enough for OpenRouter free tier to recover.
    """
    wait_times = [5, 15, 30, 60]
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["rate_limit", "429", "quota", "too many"]):
                wait = wait_times[min(attempt, len(wait_times) - 1)]
                logger.warning(f"Rate limit hit, waiting {wait}s before retry (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded due to rate limiting. Try again in a minute.")


def _build_messages_for_llm(system_content, messages, action_instruction):
    """Last message must be HumanMessage (Gemini compatibility)."""
    lc_messages = [SystemMessage(content=system_content)]
    history = []
    for m in messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history.append(AIMessage(content=m["content"]))

    if history and isinstance(history[-1], HumanMessage):
        history[-1] = HumanMessage(
            content=f"{history[-1].content}\n\n[TASK]: {action_instruction}"
        )
    else:
        history.append(HumanMessage(content=f"[TASK]: {action_instruction}"))

    lc_messages.extend(history)
    return lc_messages


def _classify_intent(messages: list[dict], llm) -> str:
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages[-6:]
    )
    prompt = [
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=f"Conversation:\n{history_text}\n\nClassify:"),
    ]
    raw = _invoke_with_retry(llm, prompt)
    # Extract just the first word (some models add explanation)
    intent = raw.strip().upper().split()[0].rstrip(".,:")
    valid = {"CLARIFY", "RECOMMEND", "REFINE", "COMPARE", "OUT_OF_SCOPE"}
    if intent not in valid:
        logger.warning(f"Unexpected intent '{intent}', defaulting to CLARIFY")
        return "CLARIFY"
    logger.info(f"Intent: {intent}")
    return intent


# ── Main agent ────────────────────────────────────────────────────────────────

def run_agent(messages: list[dict]) -> dict:
    llm = _get_llm()
    intent = _classify_intent(messages, llm)

    search_query = _build_search_query(messages)
    retrieved = retriever.search(query=search_query, k=10)
    catalog_context = retriever.format_for_prompt(retrieved)

    if intent == "COMPARE":
        all_catalog = retriever.get_all()
        extra = "\n".join(
            f"- {a['name']} ({a.get('test_type','')}): {a['description'][:120]}"
            for a in all_catalog
        )
        catalog_context = f"FULL CATALOG:\n{extra}\n\nTOP MATCHES:\n{catalog_context}"

    action_instruction = ACTION_MAP.get(intent, CLARIFY_INSTRUCTION)
    system_content = MAIN_SYSTEM.format(catalog_context=catalog_context)
    lc_messages = _build_messages_for_llm(system_content, messages, action_instruction)

    try:
        raw = _invoke_with_retry(llm, lc_messages)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return {"reply": "Temporary issue, please try again.", "recommendations": [], "end_of_conversation": False}

    try:
        parsed = _safe_json_parse(raw)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"JSON parse error: {e}")
        return {"reply": "Could you rephrase your requirement?", "recommendations": [], "end_of_conversation": False}

    recommendations = _sanitize_recommendations(parsed.get("recommendations", []))
    if intent in ("CLARIFY", "COMPARE", "OUT_OF_SCOPE"):
        recommendations = []

    return {
        "reply": parsed.get("reply", ""),
        "recommendations": recommendations,
        "end_of_conversation": bool(parsed.get("end_of_conversation", False)),
    }