"""
Puter AI LLM integration for structured extraction and RAG answer generation.
Calls Puter's REST API directly with aiohttp (no SDK needed).
Async-only. Falls back gracefully by returning empty dict / None on any failure.
"""

import os
import json
import logging
import aiohttp
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PUTER_USERNAME = os.getenv("PUTER_USERNAME")
PUTER_PASSWORD = os.getenv("PUTER_PASSWORD")
PUTER_MODEL = os.getenv("PUTER_MODEL", "gpt-4o")

PUTER_LOGIN_URL = "https://puter.com/login"
PUTER_API_BASE = "https://api.puter.com"


# ---------- EXTRACTION PROMPT ----------
EXTRACTION_SYSTEM_PROMPT = (
    "You extract structured data from logistics documents and return JSON only. "
    "No explanation."
)

EXTRACTION_USER_PROMPT = """You are an information extraction engine.

Extract the following fields EXACTLY as they appear in the document.
The document is a logistics/freight/shipping document.

Rules:
- Use ONLY the document text below
- Do NOT infer or guess values
- If a value is missing or unclear, return null
- Return ONLY valid JSON — no markdown fencing, no explanation
- For dates, use ISO 8601 format (YYYY-MM-DDTHH:MM:SS) if time is available, otherwise YYYY-MM-DD

JSON schema:
{{
  "shipment_id": string or null,
  "shipper": string or null,
  "consignee": string or null,
  "pickup_datetime": string or null,
  "delivery_datetime": string or null,
  "equipment_type": string or null,
  "mode": string or null,
  "rate": string or null,
  "currency": string or null,
  "weight": string or null,
  "carrier_name": string or null
}}

Document text:
\"\"\"
{document_text}
\"\"\"
"""


# ---------- QA PROMPT ----------
QA_SYSTEM_PROMPT = (
    "You are a helpful logistics document assistant. "
    "Answer questions using ONLY the provided context. "
    "If the context does not contain the answer, say 'Not found in document'. "
    "Be concise and factual. Do not make up information."
)

QA_USER_PROMPT = """Context from the document:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer:"""


# ---------- AUTH CACHE ----------
_cached_token = None

# Puter requires browser-like headers for login
_BROWSER_HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Origin": "https://puter.com",
    "Referer": "https://puter.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


async def _get_token():
    """Login to Puter and return a bearer token. Caches across calls."""
    global _cached_token
    if _cached_token:
        return _cached_token

    async with aiohttp.ClientSession(headers=_BROWSER_HEADERS) as session:
        async with session.post(
            PUTER_LOGIN_URL,
            json={"username": PUTER_USERNAME, "password": PUTER_PASSWORD},
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"Puter login failed: HTTP {resp.status} — {body[:200]}")
                return None
            data = await resp.json()
            if data.get("proceed"):
                _cached_token = data.get("token")
                logger.info("Puter login successful")
            else:
                logger.error(f"Puter login rejected: {data}")
            return _cached_token


# ---------- SHARED CLIENT HELPER ----------
async def _call_puter_chat(system_prompt: str, user_prompt: str):
    """
    Calls Puter's REST API directly for chat completion.
    Returns the content string. Returns None on any error.
    """
    if not PUTER_USERNAME or not PUTER_PASSWORD:
        logger.warning("Puter credentials not set in .env")
        return None

    try:
        token = await _get_token()
        if not token:
            return None

        payload = {
            "interface": "puter-chat-completion",
            "driver": "openai-completion",
            "method": "complete",
            "args": {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "model": PUTER_MODEL,
                "stream": False,
            },
        }

        api_headers = {**_BROWSER_HEADERS, "Authorization": f"Bearer {token}"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{PUTER_API_BASE}/drivers/call",
                json=payload,
                headers=api_headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"Puter API error: HTTP {resp.status} — {body[:200]}")
                    # Token might have expired, clear cache
                    global _cached_token
                    _cached_token = None
                    return None

                data = await resp.json()
                content = _extract_content(data)
                if content:
                    return content.strip()
                logger.error(f"Could not extract content from response: {json.dumps(data)[:500]}")
                return None

    except Exception as e:
        logger.error(f"Puter LLM call failed: {e}")
        return None


def _extract_content(data):
    """
    Extract text content from Puter API response.
    Handles multiple response formats (OpenAI, Claude, raw).
    """
    if isinstance(data, str):
        return data

    if not isinstance(data, dict):
        return str(data) if data else None

    # Puter wraps response in result.message.content (Claude-style)
    try:
        msg = data["result"]["message"]["content"]
        if isinstance(msg, list):
            return " ".join(
                block.get("text", "") for block in msg
                if isinstance(block, dict)
            )
        return str(msg)
    except (KeyError, TypeError, IndexError):
        pass

    # OpenAI-style: choices[0].message.content
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, TypeError, IndexError):
        pass

    # Nested: response.result.message.content
    resp = data.get("response", data.get("result", {}))
    if isinstance(resp, dict):
        try:
            msg = resp.get("result", resp).get("message", {}).get("content")
            if msg:
                return str(msg) if not isinstance(msg, list) else " ".join(
                    b.get("text", "") for b in msg if isinstance(b, dict)
                )
        except (AttributeError, TypeError):
            pass

    # Puter text field
    if "text" in data:
        return data["text"]

    # message.content at top level
    try:
        return data["message"]["content"]
    except (KeyError, TypeError):
        pass

    return None


# ---------- FULL NULL TEMPLATE ----------
EMPTY_EXTRACTION = {
    "shipment_id": None,
    "shipper": None,
    "consignee": None,
    "pickup_datetime": None,
    "delivery_datetime": None,
    "equipment_type": None,
    "mode": None,
    "rate": None,
    "currency": None,
    "weight": None,
    "carrier_name": None,
}


# ---------- PUBLIC API ----------
async def puter_extract_structured_fields(document_text: str) -> dict:
    """
    Use putergenai LLM to extract all 11 structured fields.
    Returns a dict with all keys present (nulls for missing).
    Returns empty dict on total failure (caller should fall back to regex).
    """
    user_prompt = EXTRACTION_USER_PROMPT.format(
        document_text=document_text[:6000]
    )
    raw = await _call_puter_chat(EXTRACTION_SYSTEM_PROMPT, user_prompt)

    if not raw:
        return {}

    # Strip markdown code fences if the LLM wraps the JSON
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])

    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}") + 1
        parsed = json.loads(cleaned[start:end])
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse LLM extraction JSON: {e}")
        return {}

    # Merge with template so all 11 keys are always present
    result = dict(EMPTY_EXTRACTION)
    for key in EMPTY_EXTRACTION:
        if key in parsed and parsed[key] is not None:
            result[key] = str(parsed[key])

    return result


async def puter_generate_answer(question: str, context_chunks: list) -> str:
    """
    Use putergenai LLM to generate a natural language answer from retrieved chunks.
    Returns the answer string, or None on failure (caller falls back to raw chunk).
    """
    context = "\n---\n".join(chunk["text"] for chunk in context_chunks[:3])
    user_prompt = QA_USER_PROMPT.format(context=context, question=question)
    return await _call_puter_chat(QA_SYSTEM_PROMPT, user_prompt)
