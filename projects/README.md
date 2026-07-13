# Retail AI Portfolio — Abhishek Mishra

**Analytics & Decision Science Leader · 0-to-1 AI/Data Decision Platforms**

Two compact, working, public demos of how I think about — and lead the building
of — production AI/data systems: governed scope, honest limits, evaluation
discipline, and business-facing design. Both are public-data analogues of
production platforms I have led professionally in enterprise retail. Because I
cannot share proprietary data or systems, these reproduce the same design
philosophy on public data.

| App | Live demo | What it demonstrates |
|---|---|---|
| **🛒 Retail Assist** | https://retail-ai-portfolio-ab.streamlit.app/ | Governed natural-language analytics: NL question → validated routing → static SQL templates → correct metric logic (AOV done right) → visible query plan → safe refusal of unsupported questions → 50-question eval benchmark → optional access-gated LLM mode that words answers but never touches the math |
| **🛍️ Retail Rec** | https://retail-ai-portfolio-recs-ab.streamlit.app/ | "Customers Also Bought" recommendations: co-purchase mining (confidence / support / lift) → YAML-configured scoring → a visibly separate business-rules layer (inventory, eligibility, category coherence, MMR diversity) → labeled fallbacks for cold-start coverage → honest reason codes on every recommendation |

## Shared design philosophy

- **Governed scope beats broad unreliability.** Each app supports a small set
  of things and does them correctly every time. Unsupported requests are
  refused with an honest reason, never guessed.
- **Statistics and business rules live in separate layers.** Learned or
  statistical relevance first; merchant-style constraints applied after,
  independently changeable.
- **The LLM is a language layer, never a dependency.** Retail Assist runs fully
  deterministic by default; the optional LLM re-words answers and parses intent
  against strict allowlists, and any failure falls back silently. Retail Rec
  uses no LLM at all.
- **Evaluation before claims.** Assist ships a labeled 50-question routing
  benchmark; Rec ships offline coverage/quality metrics — and neither app
  claims online lift, A/B results, or revenue impact, because public data
  cannot support such claims.
- **Honesty as a feature.** Synthetic demo fields are labeled in the UI, known
  artifacts are disclosed, and each app has a Deliberate Limits page stating
  what it does not do — by design.

## Data disclosure

Transactions: the public **UCI Online Retail** dataset (a UK online wholesaler,
Dec 2010 – Dec 9, 2011). Category, margin, channel, promotion, customer-segment,
fulfillment, inventory, and eligibility fields are synthetic demo enrichments,
deterministically generated. No employer data, code, schemas, ranking logic, or
confidential business rules are included anywhere in this repository.

## Repository map

```
projects/
  retail_assist/    app.py · src/ (semantic layer, router, SQL templates,
                    answer builder, LLM layer, evaluator) · configs/ ·
                    eval/ (labeled benchmark + results) · data/processed/
  retail_rec/       app.py · src/ (data prep, candidate mining, scoring,
                    rerank + fallbacks, reason codes, search, evaluation) ·
                    configs/ · artifacts/ (served recommendations + metrics)
```

Each project README covers its architecture, local run commands, and deployment
notes. Both apps deploy on Streamlit Community Cloud from this repo; committed
artifacts mean a fresh clone runs with just `pip install -r requirements.txt`
and the `streamlit run` command.

## Contact

Abhishek Mishra · abmishra@umich.edu ·
[LinkedIn](https://www.linkedin.com/in/abmishr4/) ·
[Portfolio](https://bumpy-accordion-b01.notion.site)
