"""
Retail Assist — query_router.py

Deterministic (Mode 1) routing: maps a natural-language question to one of
the supported intent families using transparent keyword rules, then builds a
validated query plan from the static SQL templates.

Governance rules enforced here:
  * Only allowlisted intents/metrics/dimensions ever reach SQL.
  * Questions touching explicitly unsupported domains (forecasting,
    causality, real-time, inventory, ...) are refused with an honest reason.
  * Low-confidence matches (< 0.4) are refused rather than guessed.

The optional LLM mode produces the SAME route structure (validated against
the same allowlists) — it changes how the question is parsed, never what
the system is allowed to compute.
"""

import re

from semantic_layer import SemanticLayer
from sql_templates import build_plan

CONFIDENCE_FLOOR = 0.4

# ---------------------------------------------------------------------------
# Signal vocabularies (all matching is on the normalized question)
# ---------------------------------------------------------------------------
UNSUPPORTED_SIGNALS = {
    "forecasting or prediction": ["forecast", "predict", "next quarter", "next month", "next year", "will we", "projection"],
    "causality or attribution": ["why did", "why is", "why are", "reason for", "caused", "because of", "attribution", "churn"],
    "real-time data": ["real-time", "real time", "right now", "today's", "live sales", "currently", "as of now"],
    "inventory or stock levels": ["inventory", "stock level", "in stock", "out of stock", "warehouse"],
    "competitor or market data": ["competitor", "market share", "versus amazon", "industry average"],
    "customer complaints or reviews": ["complaint", "review", "rating", "csat", "nps"],
    "employee or supplier performance": ["employee", "staff performance", "supplier", "vendor performance", "labor cost"],
    "customer lifetime value": ["lifetime value", "cltv", " ltv", "clv"],
    "weather or external factors": ["weather", "rainfall", "temperature", "holiday impact"],
    "product images": ["image", "photo", "picture"],
}

METRIC_SIGNALS = {
    "aov": ["aov", "average order value", "average order", "basket size"],
    "gross_margin": ["gross margin", "margin dollars", "total margin"],
    "margin_rate": ["margin rate", "margin %", "margin percent", "profitability rate"],
    "units": ["units", "quantity", "quantities", "volume sold", "items sold", "unit sold"],
    "orders": ["orders", "order count", "invoices", "transactions", "number of orders", "order volume"],
    "revenue": ["revenue", "sales", "sell", "sold", "turnover", "gmv"],
}

DIMENSION_SIGNALS = {
    "product_category": ["category", "categories"],
    "customer_segment": ["segment", "segments", "customer group"],
    "channel": ["channel", "channels", "web vs", "mobile app", "marketplace"],
    "promo_flag": ["promo", "promotion", "promotional", "discount", "non-promo"],
    "fulfillment_type": ["fulfillment", "fulfilment", "pickup", "locker", "ship to home", "delivery type"],
    "country": ["country", "countries", "geograph", "region", "united kingdom", "international"],
    "product": ["product", "products", "item", "items", "sku", "seller", "sellers"],
    "month": ["month", "monthly", "trend", "over time", "by week", "timeline", "time series"],
}

TOP_SIGNALS = ["top", "best", "highest", "most", "largest", "biggest", "leading", "drove", "drive"]
COMPARE_SIGNALS = ["vs", "versus", "compare", "compared", "prior month", "previous month", "month over month", "mom"]
LAST_MONTH_SIGNALS = ["last month", "latest month", "most recent month"]
HELP_SIGNALS = ["what can you", "supported question", "what do you support", "what data do you have",
                "help", "capabilities", "what questions"]


def normalize(question: str) -> str:
    q = question.lower().strip()
    return re.sub(r"\s+", " ", q)


def route_question(sem: SemanticLayer, question: str) -> dict:
    """Return a route dict:
    {intent, metrics, dimensions, time_window, top_n, confidence,
     unsupported_reason (or None)}
    """
    q = normalize(question)

    # ------------------------------------------------------------------
    # 1. Help / supported-questions intent
    # ------------------------------------------------------------------
    if any(s in q for s in HELP_SIGNALS):
        return _route("supported_questions", [], [], "all_time", 0.95)

    # ------------------------------------------------------------------
    # 2. Explicitly unsupported domains — refuse honestly, never guess
    # ------------------------------------------------------------------
    for domain, signals in UNSUPPORTED_SIGNALS.items():
        if any(s in q for s in signals):
            return _route("unsupported", [], [], "all_time", 0.95,
                          reason=f"This question involves {domain}, which is outside "
                                 f"the governed demo scope. {sem.refusal_message()}")

    # ------------------------------------------------------------------
    # 3. Detect metrics and dimensions from transparent keyword signals
    # ------------------------------------------------------------------
    metrics = [m for m, sigs in METRIC_SIGNALS.items() if any(s in q for s in sigs)]
    dims = [d for d, sigs in DIMENSION_SIGNALS.items() if any(s in q for s in sigs)]

    time_window = "all_time"
    if any(s in q for s in COMPARE_SIGNALS) and "month" in dims:
        time_window = "mom"
    elif any(s in q for s in LAST_MONTH_SIGNALS):
        time_window = "last_month"

    wants_top = any(s in q for s in TOP_SIGNALS)

    # ------------------------------------------------------------------
    # 4. Pick the intent family (first match wins, most specific first)
    # ------------------------------------------------------------------
    # product_category is checked BEFORE product: "product categories"
    # matches both signals, and category is the breakdown being asked for.
    if "product_category" in dims:
        intent, dims_out = "top_categories", ["product_category"]
    elif "product" in dims and (wants_top or metrics):
        intent, dims_out = "top_products", ["product"]
    elif "aov" in metrics and not dims:
        intent, dims_out = "aov_summary", []
    elif "country" in dims:
        intent, dims_out = "country_order_volume", ["country"]
    elif "promo_flag" in dims:
        intent, dims_out = "promo_comparison", ["promo_flag"]
    elif "customer_segment" in dims:
        intent, dims_out = "customer_segment_summary", ["customer_segment"]
    elif "channel" in dims:
        intent, dims_out = "channel_summary", ["channel"]
    elif "fulfillment_type" in dims:
        intent, dims_out = "fulfillment_mix", ["fulfillment_type"]
    elif "month" in dims or time_window == "mom":
        intent, dims_out = "revenue_trend", ["month"]
    elif metrics:
        # A bare metric question ("what is revenue?") -> overall summary
        intent, dims_out = "overall_summary", []
    else:
        return _route("unsupported", [], [], "all_time", 0.2,
                      reason=sem.refusal_message())

    # Sensible metric defaults per intent when none stated explicitly.
    if not metrics:
        metrics = ["revenue", "orders"]
    if intent == "aov_summary" and "orders" not in metrics:
        metrics = metrics + ["orders"]
    # AOV-by-dimension questions keep AOV even inside other intents.
    metrics = list(dict.fromkeys(metrics))          # dedupe, keep order

    # ------------------------------------------------------------------
    # 5. Transparent confidence: base 0.5, +0.15 per independent signal
    # ------------------------------------------------------------------
    signal_count = min(len(metrics) + len(dims) + (1 if wants_top else 0), 3)
    confidence = round(min(0.5 + 0.15 * signal_count, 0.95), 2)
    if confidence < CONFIDENCE_FLOOR:
        return _route("unsupported", [], [], "all_time", confidence,
                      reason=sem.refusal_message())

    return _route(intent, metrics, dims_out, time_window, confidence,
                  top_n=10 if intent in ("top_products", "top_categories") else None)


def _route(intent, metrics, dimensions, time_window, confidence,
           reason=None, top_n=None) -> dict:
    return {
        "intent": intent,
        "metrics": metrics,
        "dimensions": dimensions,
        "time_window": time_window,
        "top_n": top_n,
        "confidence": confidence,
        "unsupported_reason": reason,
    }


# ---------------------------------------------------------------------------
# Route -> static SQL template mapping. Every supported intent resolves to
# one of the four pre-written skeletons in sql_templates.py.
# ---------------------------------------------------------------------------
INTENT_TEMPLATE = {
    "revenue_trend":            ("trend_by_month",      ["month"]),
    "top_products":             ("top_n_products",      ["product"]),
    "top_categories":           ("metric_by_dimension", ["product_category"]),
    "aov_summary":              ("overall_summary",     []),
    "country_order_volume":     ("metric_by_dimension", ["country"]),
    "promo_comparison":         ("metric_by_dimension", ["promo_flag"]),
    "customer_segment_summary": ("metric_by_dimension", ["customer_segment"]),
    "channel_summary":          ("metric_by_dimension", ["channel"]),
    "fulfillment_mix":          ("metric_by_dimension", ["fulfillment_type"]),
    "overall_summary":          ("overall_summary",     []),
}


def plan_from_route(sem: SemanticLayer, route: dict) -> dict | None:
    """Build a validated SQL plan for a supported route. None if unsupported."""
    if route["intent"] in ("unsupported", "supported_questions"):
        return None
    template, dims = INTENT_TEMPLATE[route["intent"]]
    try:
        return build_plan(
            sem, template,
            metrics=route["metrics"],
            dimensions=dims,
            time_window=route["time_window"],
            top_n=route["top_n"] or 10,
        )
    except ValueError:
        # Anything the semantic layer rejects becomes a safe refusal.
        route["intent"] = "unsupported"
        route["unsupported_reason"] = sem.refusal_message()
        return None
