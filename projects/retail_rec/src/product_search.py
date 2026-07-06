"""
Retail Rec — product_search.py

Simple, inspectable product search over the product lookup:
  * exact stock_code match wins immediately
  * otherwise token-overlap: any query token matching any description
    token counts; results ranked by matched-token count, then product
    popularity (order_count)

Deliberately NOT fuzzy edit distance — token overlap is predictable,
explainable, and good enough for a 3.9k-product catalog.
"""

import re

import pandas as pd

_TOKEN_RE = re.compile(r"[a-z0-9]+")

NO_RESULTS_HINT = ("No results. Try a shorter word like BAG, HEART, MUG, "
                   "BOX, RED, WHITE, SET, or LUNCH.")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(str(text).lower()))


def search_products(lookup: pd.DataFrame, query: str,
                    max_results: int = 25) -> pd.DataFrame:
    """Return matching products (possibly empty), best matches first."""
    q = str(query).strip()
    if not q:
        return lookup.head(0)

    # 1. Exact stock_code match (case-insensitive)
    code_hit = lookup[lookup["stock_code"].str.lower() == q.lower()]
    if len(code_hit):
        return code_hit.head(max_results)

    # 2. Token overlap against descriptions
    q_tokens = _tokens(q)
    if not q_tokens:
        return lookup.head(0)

    desc_tokens = lookup["description"].map(_tokens)
    matches = desc_tokens.map(lambda t: len(q_tokens & t))
    hits = lookup.assign(_matched=matches)
    hits = hits[hits["_matched"] > 0]
    hits = hits.sort_values(["_matched", "order_count"],
                            ascending=[False, False])
    return hits.drop(columns=["_matched"]).head(max_results)
