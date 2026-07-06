"""
Retail Rec — reason_codes.py

Every recommendation carries a plain-language reason code — the customer-
value layer. Codes are HONEST: they scale with the actual evidence, and
thin evidence says so out loud rather than dressing itself up.
"""


def assign_reason(strategy: str, co_purchase_count, confidence, lift,
                  same_category: bool) -> str:
    if strategy == "Category popularity fallback":
        return ("Category popularity fallback due to sparse direct "
                "co-purchase history.")
    if strategy == "Global popularity fallback":
        return "Global popularity fallback due to limited product history."

    # Direct co-purchase — graded by evidence strength.
    co = int(co_purchase_count or 0)
    conf = float(confidence or 0)
    lft = float(lift or 0)

    if co >= 20 and conf >= 0.20:
        return "Frequently bought together in the same basket."
    if lft >= 10 and co >= 5:
        return "Strong co-purchase lift versus baseline popularity."
    if same_category and co >= 3:
        return ("Commonly bought by customers purchasing similar "
                "category items.")
    if co >= 5:
        return "Repeatedly co-purchased across customer baskets."
    return ("Co-purchased in a small number of baskets — limited "
            "evidence, ranked accordingly.")
