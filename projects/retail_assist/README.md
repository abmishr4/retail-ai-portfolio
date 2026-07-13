# Retail Assist

A **governed natural-language retail analytics assistant** — ask a business
question in plain English and get a correct, governed answer with a fully
visible query plan, or an honest refusal when the question is outside scope.
A compact public demo of how I build production AI/data decision systems:
governed scope, correct metric logic, evaluation discipline, and the LLM as a
language layer that never touches the math.

**Live demo:** https://retail-ai-portfolio-ab.streamlit.app/
**Companion app:** Retail Rec — "Customers Also Bought" recommendations (same repo).

## What it is

A conversational assistant over a real retail dataset. You ask things like
*"Which categories drove revenue?"* or *"Compare promo vs non-promo sales"* and
get a business-language answer, a one-click routing summary (how the question
was routed, the detected intent, the confidence, the time window), and a
collapsible panel with the exact SQL, a chart, and the underlying rows. Ask
something outside the governed scope — a forecast, a causal "why", inventory —
and it refuses honestly instead of guessing.

This is a deliberately scaled-down public analogue of a production
natural-language decision platform I led professionally, which handled far
broader scope. Because that system runs on proprietary data and logic, this
version reproduces the same design philosophy — governance, evaluation, honest
limits — entirely on public data.

## What it demonstrates

- **Governed scope** — a semantic layer (`configs/metrics.yml`) defines every
  metric and dimension the app may compute. Nothing outside it executes.
- **Correct metric logic** — AOV is always `SUM(revenue) / COUNT(DISTINCT
  invoice)`; weighted ratios are computed at query time, never averaged across
  groups.
- **No dynamic SQL** — questions route to four static, pre-written SQL
  templates. User text never enters SQL. The optional LLM never writes SQL.
- **Safe refusal** — forecasting, causality, real-time, and other unsupported
  domains are refused with an honest reason, not guessed. Unsupported coverage
  is a first-class response type.
- **Evaluation discipline** — a 50-question labeled benchmark (35 supported /
  15 unsupported) scores routing, metric selection, dimension selection,
  time-window resolution, refusal behavior, and overall semantic correctness.
- **LLM as language layer only** — optional, access-gated Gemini mode parses
  intent into allowlist-validated JSON and re-words answers. Any failure falls
  back silently to deterministic mode. No key is needed for the app to be fully
  functional.

## Key design decisions (and the failure each one guards against)

Each of these is a deliberate call, enforced in code — not a convention.

| Decision | Why | What it prevents |
|---|---|---|
| **AOV / margin rate are weighted ratios, computed at query time** | Averaging a ratio across categories or days is silently wrong | The classic aggregation error (e.g. an implausibly high "average order value" that erodes analyst trust) |
| **Refusal is a first-class response** | A narrow, correct system beats a broad, unreliable one | Confident hallucination at the data boundary — the app says "I can't answer that" instead of improvising a number |
| **User text never reaches SQL** | Retrieval flows only through four validated templates | Prompt-injection and the unreliability of open text-to-SQL |
| **The LLM is a language layer, never a dependency** | Deterministic mode is both the default and the fallback | Outages, quota limits, retired models, or hallucinated numbers taking the app down — the math is always deterministic |
| **Wording rules live in code, not the prompt** | The answer builder enforces safety rules directly | Causal language, trend claims on a single data point, or any assertion not present in the returned rows |

## Data and honesty

Transaction data is the public **UCI Online Retail** dataset (a UK online
wholesaler, Dec 2010 – Dec 9, 2011; currency GBP). `product_category`,
`margin_rate`, `channel`, `promo_flag`, `customer_segment`, and
`fulfillment_type` are **synthetic demo enrichments**, deterministically
generated (stable-hash, no randomness) by `src/data_prep.py`. No employer data,
employer code, or confidential business logic is included. No production
performance claims are made by this demo.

## Architecture

Every question passes through the same governed pipeline — the LLM never
reasons over raw data, only over validated outputs.

```
question
  └─ route (keyword rules | optional LLM → validated JSON)
       └─ validate against semantic-layer allowlists
            └─ static SQL template (1 of 4) — user text never enters SQL
                 └─ DuckDB over pre-aggregated parquet (built offline)
                      └─ deterministic answer builder (safety rules in code)
                           └─ optional LLM re-wording (same rules, silent fallback)
```

The Streamlit app is a conversational interface over this pipeline: threaded
chat, one-click example prompts, per-answer routing transparency, and a
collapsible query-plan / chart / data panel on every response.

## Run locally

```bash
pip install -r requirements.txt
# one-time: put the UCI "Online Retail.xlsx" in projects/retail_assist/data/raw/
python projects/retail_assist/src/data_prep.py
python projects/retail_assist/src/evaluator.py          # regenerates eval results
streamlit run projects/retail_assist/app.py
```

The processed parquet artifacts are committed, so cloning and running the
`streamlit run` command alone is enough to use the app.

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
metrics, dimensions, time window, and refusal behavior. `src/evaluator.py`
routes each through the real pipeline and scores routing accuracy,
metric-selection accuracy, dimension-selection accuracy, time-window accuracy,
unsupported-refusal accuracy, and overall semantic correctness (all criteria at
once, including successful execution). Point-in-time results live in
`eval/eval_results.csv` and render in the app's Evaluation Results tab.

## Deliberate limits

Open SQL generation, free-text metric computation, forecasting, causal claims,
and real-time data are **out of scope by design** — the point is a narrow,
correct, honest system. The Deliberate Limits tab in the app states this to
every visitor.
