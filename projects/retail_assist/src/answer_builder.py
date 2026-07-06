"""
Retail Assist — answer_builder.py

Turns a query result (DataFrame + plan metadata) into a short, honest,
business-language answer. This is the DETERMINISTIC wording path; the
optional LLM answer-formatter obeys the same rules and falls back to this.

Safety rules enforced (in code, not by convention):
  1. Zero rows        -> fixed "no data" message.
  2. One row          -> plain statement of values. No trend language.
  3. Trend language   -> only when 2+ comparable time periods exist.
  4. Driver language  -> only when the result contains a breakdown dimension.
  5. Causal language  -> never. Only "associated with" / observational wording.
  6. Nothing outside the dataframe is ever asserted.
"""

import pandas as pd

from semantic_layer import SemanticLayer

# The dataset is a UK online retailer, so currency is GBP.
CURRENCY = "\u00a3"  # £


def fmt(sem: SemanticLayer, metric: str, value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    kind = sem.metric_format(metric)
    if kind == "currency":
        return f"{CURRENCY}{value:,.2f}"
    if kind == "percent":
        return f"{value * 100:.1f}%"
    if kind == "integer":
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _promo_label(v) -> str:
    # df.iloc[row] can upcast ints to float (1 -> 1.0), so compare numerically.
    try:
        return "Promo" if float(v) == 1 else "Non-promo"
    except (TypeError, ValueError):
        return "Promo" if str(v).strip().lower() in ("1", "true", "promo") else "Non-promo"


def build_answer(sem: SemanticLayer, route: dict, plan: dict | None,
                 df: pd.DataFrame | None) -> str:
    # ------------------------------------------------------------------
    # Refusals and help
    # ------------------------------------------------------------------
    if route["intent"] == "unsupported":
        return route["unsupported_reason"] or sem.refusal_message()

    if route["intent"] == "supported_questions":
        return (
            "I can answer questions about these governed metrics: "
            f"{', '.join(sem.allowed_metrics())} — broken down by "
            f"{', '.join(d for d in sem.allowed_dimensions() if d != 'invoice_date')}. "
            "Examples: 'What was revenue by month?', 'Top products by gross margin', "
            "'AOV by country', 'Compare promo vs non-promo sales'. "
            "I refuse questions outside this scope rather than guessing."
        )

    # ------------------------------------------------------------------
    # Rule 1: zero rows
    # ------------------------------------------------------------------
    if df is None or len(df) == 0:
        return "No data available for this question and time window."

    metrics = plan["metrics_used"]
    window = plan["time_window"]
    dim = plan["dimensions_used"][0] if plan["dimensions_used"] else None

    # ------------------------------------------------------------------
    # Rule 2: single row -> plain statement, no trend language
    # ------------------------------------------------------------------
    if len(df) == 1:
        row = df.iloc[0]
        parts = [f"{m.replace('_', ' ')} was {fmt(sem, m, row[m])}" for m in metrics]
        return f"For {window}: " + "; ".join(parts) + "."

    # ------------------------------------------------------------------
    # Rule 3: time series -> trend language allowed (2+ periods present)
    # ------------------------------------------------------------------
    if dim == "month":
        m = metrics[0]
        first, last = df.iloc[0], df.iloc[-1]
        change = (last[m] - first[m]) / first[m] * 100 if first[m] else None
        direction = "higher" if last[m] >= first[m] else "lower"
        answer = (
            f"Across {len(df)} months ({first['month']} to {last['month']}), "
            f"{m.replace('_', ' ')} went from {fmt(sem, m, first[m])} to "
            f"{fmt(sem, m, last[m])} — {abs(change):.1f}% {direction} at the end "
            f"of the window than the start."
        )
        if len(df) == 2:
            answer = (
                f"{m.replace('_', ' ').capitalize()} was {fmt(sem, m, first[m])} in "
                f"{first['month']} and {fmt(sem, m, last[m])} in {last['month']} — "
                f"{abs(change):.1f}% {direction} month over month."
            )
        peak = df.loc[df[m].idxmax()]
        answer += f" The highest single month was {peak['month']} at {fmt(sem, m, peak[m])}."
        return answer

    # ------------------------------------------------------------------
    # Rule 4: breakdown by a dimension -> driver language allowed
    # ------------------------------------------------------------------
    m = metrics[0]
    label_col = df.columns[0]
    top = df.iloc[0]
    top_label = _promo_label(top[label_col]) if dim == "promo_flag" else str(top[label_col])

    if dim == "promo_flag" and len(df) == 2:
        other = df.iloc[1]
        other_label = _promo_label(other[label_col])
        parts = []
        for mm in metrics:
            parts.append(
                f"{mm.replace('_', ' ')}: {top_label} {fmt(sem, mm, top[mm])} vs "
                f"{other_label} {fmt(sem, mm, other[mm])}"
            )
        return (f"Comparing promo and non-promo orders for {window} — "
                + "; ".join(parts)
                + ". (Promo fields are synthetic demo enrichments.)")

    share = ""
    if sem.metrics[m]["aggregation_rule"] == "sum":
        total = df[m].sum()
        if total:
            share = f", {top[m] / total * 100:.1f}% of the total shown"

    answer = (
        f"For {window}, the largest observed component by "
        f"{m.replace('_', ' ')} was {top_label} at {fmt(sem, m, top[m])}{share}."
    )
    if len(df) > 1:
        second = df.iloc[1]
        second_label = _promo_label(second[label_col]) if dim == "promo_flag" else str(second[label_col])
        answer += f" Next was {second_label} at {fmt(sem, m, second[m])}."
    answer += f" {len(df)} {dim.replace('_', ' ')} groups returned in total."
    return answer
