# Retail Rec

A **"Customers Also Bought" recommendation experience** built on transparent
co-purchase statistics and visible business rules — a compact public demo of
how I build production recommendation systems: candidate generation, scoring,
reranking, coverage discipline, and honest reason codes, with learned/statistical
relevance kept in a separate layer from merchant business rules.

**Live demo:** https://retail-ai-portfolio-recs-ab.streamlit.app/
**Companion app:** Retail Assist — governed natural-language analytics (same repo).

## What it is

Search a real 3,922-product catalog (UCI Online Retail), pick a product, and
see its top-20 recommendations as cards — each carrying its co-purchase count,
confidence, lift, final score, a strategy badge, a score bar, and a
plain-language reason code. Business rules (inventory, eligibility, category
coherence, low-confidence penalties, near-duplicate diversity) are applied in a
visibly separate layer **after** statistical scoring, and coverage gaps are
filled by clearly labeled fallbacks. You can also jump into any recommendation's
own recommendations to browse the graph.

This is a deliberately scaled-down public analogue of a production
recommendation platform I led professionally. Retail Rec does not claim online
business lift — it demonstrates production-style recommendation-system thinking:
candidate generation, scoring, reranking, fallback handling, coverage
discipline, reason codes, and transparent business rules.

## Key design decisions (and what each one buys)

Each of these is a deliberate call, enforced in the offline pipeline — not a
convention.

| Decision | Why | What it buys |
|---|---|---|
| **Statistics and business rules live in strictly separate layers** | Relevance is learned/statistical; inventory, eligibility, and merchandising are business policy | Rules can change (a product goes out of stock) without recomputing or retraining the relevance model |
| **Coverage is a first-class metric; fallbacks are labeled, never disguised** | A recommender that only fires for popular anchors doesn't serve the catalog | 100% product coverage with full 20-item lists, and a visitor always knows when a row is a popularity fill vs. real co-purchase evidence |
| **Reason codes are graded by actual evidence** | Thin evidence should say so, not dress itself up | Trust — a pairing seen in two baskets is labeled "limited evidence, ranked accordingly," not presented like a strong signal |
| **Lift and count are log-scaled before normalization** | Both are heavy-tailed; a single-basket pair can show lift in the hundreds | Extreme outliers don't flatten every real signal to zero under min-max |
| **Near-duplicates are diversified with an MMR penalty** | Five near-identical cakestands is a poor experience | Colorway/variant diversity without killing legitimately similar items |

## How it works

```
cleaned baskets (16,217 mined; wholesale baskets >50 items skipped)
  └─ candidate generation — itertools.combinations per basket, Counter
       aggregation (no item-item matrix, no crosstab, no cross-join)
       → co_purchase_count, confidence, support, lift per directional pair
  └─ scoring — YAML weights over normalized components; lift and count
       log-scaled first (heavy-tailed)
  └─ business rules — out-of-stock and ineligible items hard-removed,
       same-category boost, low-confidence penalty, MMR diversity
       (all knobs in YAML)
  └─ fallback fill — category popularity, then global popularity, to a
       full top-20 per product; every fill labeled
  └─ reason codes — honest, evidence-graded plain language on every row
```

The math:

- **confidence** = co_purchase_count / anchor_basket_count —
  `P(recommended in basket | anchor in basket)`
- **support** = co_purchase_count / total_baskets
- **lift** = confidence / (recommended_basket_count / total_baskets) —
  how much the pairing beats the item's baseline popularity. Lift on 1–2
  baskets is noisy, so evidence volume is scored alongside it and always
  displayed next to it.

The Streamlit app performs no computation at startup — it reads only
`product_lookup.parquet`, `final_recommendations.parquet`, and `metrics.json`,
all built offline by the `src/` pipeline.

## Data and honesty

Transactions: public **UCI Online Retail** dataset (UK online wholesaler, Dec
2010 – Dec 9, 2011). `product_category`, `synthetic_price_bucket`,
`synthetic_margin_rate`, `synthetic_in_stock`, and `synthetic_eligibility_flag`
are **synthetic demo enrichments**, deterministically generated (stable-hash, no
randomness). No employer data, code, ranking logic, or confidential business
rules are included. No online lift, A/B-test, or revenue claims are made.

## Offline metrics (from `artifacts/metrics.json`)

100% product coverage (3,922/3,922 with full 20-rec lists) · 18.5 direct
recommendations per product on average · 7.5% fallback rate · median direct
lift 23.7 · rule activity fully logged (out-of-stock and ineligible removals,
boosts, penalties). Regenerate with `python src/evaluate.py`.

## Run locally

```bash
pip install -r requirements.txt
# one-time: put the UCI "Online Retail.xlsx" in projects/retail_rec/data/raw/
python projects/retail_rec/src/data_prep.py
python projects/retail_rec/src/build_candidates.py
python projects/retail_rec/src/rerank.py
python projects/retail_rec/src/evaluate.py
streamlit run projects/retail_rec/app.py
```

The committed artifacts make the last command sufficient on a fresh clone.

## Deploy (Streamlit Community Cloud)

New app → repo `retail-ai-portfolio` → branch `main` → main file path
`projects/retail_rec/app.py`. No secrets, no API keys — this app is fully
offline.

## Deliberate limits

No deep learning in this MVP (co-purchase + lift + confidence + reranking
demonstrates the thinking without a model server) · no personalization
(anchor-product-based, not user-based) · small sparse dataset versus enterprise
scale · synthetic inventory/eligibility/category fields, disclosed in-app ·
keyword category labels occasionally misfire (accepted demo artifact). The
Deliberate Limits page states all of this to every visitor.
