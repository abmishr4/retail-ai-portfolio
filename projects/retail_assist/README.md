# Retail Assist

A governed natural-language retail analytics assistant — a compact public
demo of how I think about production AI/data systems.

**Live demo:** _add your Streamlit URL here_

## What it is

Ask a business question in plain English ("Which categories drove
revenue?", "Compare promo vs non-promo sales") and get a correct,
business-language answer with a fully visible query plan — or an honest
refusal when the question is outside the governed scope.

This is a deliberately scaled-down public analogue of a production
natural-language decision platform I led professionally. The production
system handled far broader scope; here the goal is a small supported set
answered correctly every time. Hiring managers can't see proprietary
systems — this shows the same design philosophy on public data.

## What it demonstrates

- **Governed scope** — a semantic layer (`configs/metrics.yml`) defines
  every metric the app may compute. Nothing outside it executes.
- **Correct metric logic** — AOV is always `SUM(revenue) / COUNT(DISTINCT
  invoice)`; weighted ratios are computed at query time, never averaged
  across groups.
- **No dynamic SQL** — questions route to four static, pre-written SQL
  templates. User text never enters SQL. The optional LLM never writes SQL.
- **Safe refusal** — forecasting, causality, real-time, and other
  unsupported domains are refused with an honest reason, not guessed.
- **Evaluation discipline** — a 50-question labeled benchmark
  (35 supported / 15 unsupported) scores routing, metric selection,
  refusal behavior, and overall semantic correctness.
- **LLM as language layer only** — optional, access-gated Gemini mode
  parses intent into allowlist-validated JSON and re-words answers.
  Any failure falls back silently to deterministic mode. No key needed
  for the app to be fully functional.

## Data and honesty

Transaction data is the public **UCI Online Retail** dataset (a UK online
wholesaler, Dec 2010 – Dec 9, 2011; currency GBP). `product_category`,
`margin_rate`, `channel`, `promo_flag`, `customer_segment`, and
`fulfillment_type` are **synthetic demo enrichments**, deterministically
generated (stable-hash, no randomness) by `src/data_prep.py`. No employer
data, employer code, or confidential business logic is included. No
production performance claims are made by this demo.

## Architecture

```
question
  └─ route (keyword rules | optional LLM → validated JSON)
       └─ validate against semantic layer allowlists
            └─ static SQL template (1 of 4) — user text never enters SQL
                 └─ DuckDB over pre-aggregated parquet (built offline)
                      └─ deterministic answer builder (safety rules in code)
                           └─ optional LLM re-wording (same rules, silent fallback)
```

## Run locally

```bash
pip install -r requirements.txt
# one-time: put the UCI "Online Retail.xlsx" in projects/retail_assist/data/raw/
python projects/retail_assist/src/data_prep.py
python projects/retail_assist/src/evaluator.py          # regenerates eval results
streamlit run projects/retail_assist/app.py
```

The processed parquet artifacts are committed, so cloning and running the
Streamlit command alone is enough to use the app.

## Deploy (Streamlit Community Cloud)

1. Fork/push this repo (public).
2. share.streamlit.io → New app → main file path
   `projects/retail_assist/app.py`.
3. Optional LLM mode: in the app's **Settings → Secrets**, add
   ```toml
   GEMINI_API_KEY = "..."
   DEMO_PIN = "..."
   ENABLE_LLM_MODE = true
   ```
   Without secrets the app runs fully in deterministic mode.

## Evaluation methodology

`eval/eval_questions.csv` holds 50 labeled questions with expected intent,
metrics, dimensions, time window, and refusal behavior.
`src/evaluator.py` routes each through the real pipeline and scores:
routing accuracy, metric-selection accuracy, dimension-selection accuracy,
time-window accuracy, unsupported-refusal accuracy, and overall semantic
correctness (all criteria at once, including successful execution).
Point-in-time results live in `eval/eval_results.csv` and render in the
app's Evaluation Results tab.

## Deliberate limits

Open SQL generation, free-text metric computation, forecasting, causal
claims, and real-time data are **out of scope by design** — the point is a
narrow, correct, honest system. The Deliberate Limits tab in the app states
this to every visitor.

## Screenshots

_add screenshots from assets/screenshots/ after deployment_
