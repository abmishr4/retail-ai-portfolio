"""
Retail Assist — llm_layer.py

OPTIONAL LLM-assisted mode (Mode 2), key-gated in the app. Uses the current
google-genai SDK (the old google-generativeai package is end-of-life).

The LLM has exactly two allowed jobs:
  JOB 1  parse_intent_llm : question -> strict JSON route, then validated
         against the SAME allowlists as deterministic mode. The LLM never
         chooses tables, writes SQL, or computes metrics.
  JOB 2  word_answer_llm  : returned dataframe rows + plan metadata ->
         concise business wording, under the same safety rules as
         answer_builder (no causal language, nothing outside the data).

Failure policy: ANY error (bad key, quota, network, invalid JSON, allowlist
violation, retired model name) returns None, and the caller falls back to
deterministic mode. The LLM is a language layer, never a dependency.

The API key is supplied by app.py from st.secrets — never hardcoded, never
read from the environment here.
"""

import json

from semantic_layer import SemanticLayer

# Single place to update when Google retires/renames models.
# gemini-2.5-flash-lite is current and free-tier eligible; a newer
# flash-lite generation can be swapped in here with no other changes.
GEMINI_MODEL = "gemini-2.5-flash-lite"

ALLOWED_INTENTS = [
    "revenue_trend", "top_products", "top_categories", "aov_summary",
    "country_order_volume", "promo_comparison", "customer_segment_summary",
    "channel_summary", "fulfillment_mix", "overall_summary",
    "supported_questions", "unsupported",
]
ALLOWED_TIME_WINDOWS = ["all_time", "last_month", "mom"]
CAUSAL_WORDS = ["because", "caused", "due to", "thanks to", "driven by", "as a result of"]


def _client(api_key: str):
    from google import genai   # imported lazily so deterministic mode never needs it
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# JOB 1 — intent parsing into a validated route
# ---------------------------------------------------------------------------
def parse_intent_llm(sem: SemanticLayer, question: str, api_key: str) -> dict | None:
    """Return a route dict identical in shape to query_router.route_question,
    or None on any failure (caller falls back to deterministic routing)."""
    prompt = f"""You classify retail analytics questions into a strict JSON route.
Respond with ONLY a JSON object, no prose, no markdown fences.

Allowed intents: {ALLOWED_INTENTS}
Allowed metrics: {sem.allowed_metrics()}
Allowed dimensions: {sem.allowed_dimensions()}
Allowed time_window values: {ALLOWED_TIME_WINDOWS}
  ("mom" = the question compares a month to the prior month;
   "last_month" = the question asks only about the most recent month;
   otherwise "all_time")

Rules:
- If the question needs anything outside these lists (forecasting, causality,
  real-time data, inventory, competitors, complaints, employees, suppliers,
  lifetime value, weather, images), set intent="unsupported".
- If you are not confident, set intent="unsupported" and confidence below 0.7.
- Never invent metrics or dimensions.

JSON schema:
{{"intent": "...", "metrics": ["..."], "dimensions": ["..."],
  "time_window": "...", "confidence": 0.0, "unsupported_reason": null}}

Question: {question}"""

    try:
        client = _client(api_key)
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.0},
        )
        route = json.loads(resp.text)
    except Exception:
        return None

    # ------------------------------------------------------------------
    # Validate EVERYTHING against the same allowlists as Mode 1.
    # ------------------------------------------------------------------
    try:
        if route["intent"] not in ALLOWED_INTENTS:
            return None
        route["metrics"] = [m for m in route.get("metrics", []) if m in sem.allowed_metrics()]
        route["dimensions"] = [d for d in route.get("dimensions", []) if d in sem.allowed_dimensions()]
        if route.get("time_window") not in ALLOWED_TIME_WINDOWS:
            route["time_window"] = "all_time"
        conf = float(route.get("confidence", 0))
        route["confidence"] = round(min(max(conf, 0.0), 1.0), 2)
        if route["confidence"] < 0.7 and route["intent"] != "unsupported":
            route["intent"] = "unsupported"
        if route["intent"] == "unsupported" and not route.get("unsupported_reason"):
            route["unsupported_reason"] = sem.refusal_message()
        if route["intent"] not in ("unsupported", "supported_questions") and not route["metrics"]:
            route["metrics"] = ["revenue", "orders"]
        route["top_n"] = 10 if route["intent"] in ("top_products", "top_categories") else None
        route.setdefault("unsupported_reason", None)
        return route
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JOB 2 — wording the answer from the returned data (never computing it)
# ---------------------------------------------------------------------------
def word_answer_llm(sem: SemanticLayer, question: str, plan: dict,
                    df, deterministic_answer: str, api_key: str) -> str | None:
    """Return a concise business-language answer, or None on any failure
    (caller shows the deterministic answer instead)."""
    if df is None or len(df) == 0:
        return None  # fixed no-data message comes from answer_builder

    rows_csv = df.head(15).to_csv(index=False)
    defs = {m: sem.metric_description(m) for m in plan["metrics_used"]}

    prompt = f"""You word a retail analytics answer for a business audience.
Use ONLY the data rows, plan metadata, and metric definitions below.

Hard rules:
- 2 to 4 sentences, plain business language.
- Do not add ANY external context, benchmarks, or explanations of why.
- Never use causal language (because, caused, due to, driven by).
- Trend words (up, down, grew, declined) ONLY if 2+ time periods are present.
- If the data does not answer the question, reply exactly: INSUFFICIENT
- All numbers must come from the rows. Currency is GBP (\u00a3).

Question: {question}
Time window: {plan['time_window']}
Metrics and definitions: {json.dumps(defs)}
Data rows (CSV, up to 15):
{rows_csv}

Reference wording (deterministic, known-correct): {deterministic_answer}"""

    try:
        client = _client(api_key)
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": 0.2},
        )
        text = (resp.text or "").strip()
    except Exception:
        return None

    # Output guards: refuse to display anything that breaks the rules.
    if not text or "INSUFFICIENT" in text.upper():
        return None
    if any(w in text.lower() for w in CAUSAL_WORDS):
        return None
    if len(text) > 700:
        return None
    return text
