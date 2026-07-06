"""
Retail Assist — app.py

Public portfolio demo: a governed natural-language retail analytics
assistant over public UCI data with synthetic enrichments.

Default mode is fully deterministic (no API key needed, ever). The optional
LLM-assisted mode is access-gated and only re-words answers / parses intent
— it never generates SQL, computes metrics, or adds business context.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from semantic_layer import SemanticLayer                      # noqa: E402
from query_router import route_question, plan_from_route      # noqa: E402
from sql_templates import run_plan                            # noqa: E402
from answer_builder import build_answer                       # noqa: E402
from llm_layer import parse_intent_llm, word_answer_llm, GEMINI_MODEL  # noqa: E402

EVAL_DIR = BASE_DIR / "eval"
CONTACT_EMAIL = "abmishra@umich.edu"
MAX_LLM_CALLS = 5   # courtesy limit per session (NOT a security control —
                    # the access key is the real gate)

EXAMPLES = [
    "What was revenue by month?",
    "Which categories drove revenue?",
    "Compare promo vs non-promo sales",
    "Forecast profit for next quarter",   # refused by design — the governance demo
]

st.set_page_config(page_title="Retail Assist", page_icon="🛒", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def get_semantic_layer() -> SemanticLayer:
    return SemanticLayer()


def get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def make_chart(plan, df):
    """One simple chart per answer: line for time, bar for breakdowns."""
    if df is None or len(df) < 2:
        return None
    metric = plan["metrics_used"][0]
    dim = plan["dimensions_used"][0] if plan["dimensions_used"] else None
    plot_df = df.copy()
    if dim == "promo_flag":
        plot_df["promo_flag"] = plot_df["promo_flag"].map(
            lambda v: "Promo" if float(v) == 1 else "Non-promo")
    if dim == "month":
        fig = px.line(plot_df, x="month", y=metric, markers=True)
    else:
        label_col = plot_df.columns[0]
        fig = px.bar(plot_df.head(15), x=label_col, y=metric)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title=metric.replace("_", " "))
    return fig


def answer_question(sem, question, llm_active, api_key):
    """Full pipeline for one question. Returns a dict for rendering."""
    det_route = route_question(sem, question)
    route, routed_by = det_route, "deterministic keywords"

    if llm_active and st.session_state.llm_calls < MAX_LLM_CALLS:
        llm_route = parse_intent_llm(sem, question, api_key)
        st.session_state.llm_calls += 1
        if llm_route is not None:
            route, routed_by = llm_route, f"LLM ({GEMINI_MODEL}), allowlist-validated"

    plan = plan_from_route(sem, route)
    df = run_plan(plan) if plan else None
    det_answer = build_answer(sem, route, plan, df)

    llm_answer = None
    if (llm_active and plan is not None and df is not None and len(df) > 0
            and st.session_state.llm_calls < MAX_LLM_CALLS):
        llm_answer = word_answer_llm(sem, question, plan, df, det_answer, api_key)
        st.session_state.llm_calls += 1

    return {"question": question, "route": route, "routed_by": routed_by,
            "plan": plan, "df": df, "det_answer": det_answer,
            "llm_answer": llm_answer}


def run_ask(sem, question, llm_active, api_key):
    with st.spinner("Answering..."):
        st.session_state.last_result = answer_question(
            sem, question, llm_active, api_key)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
sem = get_semantic_layer()
if "llm_calls" not in st.session_state:
    st.session_state.llm_calls = 0
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ---------------------------------------------------------------------------
# Sidebar — navigation, mode, access gate
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛒 Retail Assist")
    st.caption("Governed natural-language retail analytics — public demo")

    page = st.radio(
        "Navigate",
        ["💬 Ask Retail Assist", "❓ Supported Questions", "📖 Semantic Layer",
         "✅ Evaluation Results", "🏗️ Architecture", "⚠️ Deliberate Limits"],
        label_visibility="collapsed",
    )

    st.divider()
    use_llm = st.checkbox("Use LLM-assisted governed mode")
    pin_input = st.text_input("Demo access key", type="password",
                              disabled=not use_llm)

    api_key = get_secret("GEMINI_API_KEY")
    demo_pin = get_secret("DEMO_PIN")
    llm_enabled_flag = bool(get_secret("ENABLE_LLM_MODE", False))

    llm_active = bool(use_llm and llm_enabled_flag and api_key and demo_pin
                      and pin_input == demo_pin)

    if llm_active:
        st.success("Mode: LLM-assisted governed")
        st.caption(f"LLM calls this session: {st.session_state.llm_calls} / "
                   f"{MAX_LLM_CALLS}. At the limit, the app falls back to "
                   "deterministic mode.")
    else:
        st.info("Mode: Deterministic governed (default)")
        if use_llm:
            st.warning(
                "LLM mode is access-gated. Email me at "
                f"{CONTACT_EMAIL} to request a demo key. You are seeing "
                "deterministic mode — same numbers, same governance; the key "
                "only unlocks LLM-worded answers.")

    st.divider()
    st.caption("Data: UCI Online Retail (public), Dec 2010 – Dec 9 2011. "
               "Category, margin, channel, promo, segment, and fulfillment "
               "fields are synthetic demo enrichments.")

# ---------------------------------------------------------------------------
# Page: Ask Retail Assist
# ---------------------------------------------------------------------------
if page == "💬 Ask Retail Assist":
    st.subheader("Ask a business question")

    # Click-to-ask examples (recruiters shouldn't have to type)
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        label = ex if ex != EXAMPLES[-1] else f"{ex} 🚫"
        if col.button(label, key=f"ex_{ex}", help="Click to ask"):
            run_ask(sem, ex, llm_active, api_key)

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_input(
            "Your question",
            placeholder="e.g., Which customer segments drive revenue?")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted and question.strip():
        run_ask(sem, question.strip(), llm_active, api_key)

    result = st.session_state.last_result

    if result is None:
        st.info(
            "👋 **Welcome to Retail Assist.** Ask any business question about "
            "this retail dataset — revenue, orders, units, AOV, margin — by "
            "month, product, category, country, segment, channel, promo, or "
            "fulfillment. Click an example above or type your own. "
            "Questions outside the governed scope are refused honestly, "
            "never guessed — try the last example to see it.")
    else:
        route, plan, df = result["route"], result["plan"], result["df"]
        st.caption(f'You asked: "{result["question"]}"')

        # ----- Answer -----
        if result["llm_answer"]:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Deterministic answer** (template-worded)")
                st.success(result["det_answer"])
            with c2:
                st.markdown(f"**LLM-worded answer** ({GEMINI_MODEL})")
                st.success(result["llm_answer"])
            st.caption("Same query plan, same numbers — the LLM changes the "
                       "wording, never the math.")
        elif route["intent"] == "unsupported":
            st.error(result["det_answer"])
        else:
            st.success(result["det_answer"])

        # ----- Details, demoted to small tabs -----
        if route["intent"] != "supported_questions":
            t_chart, t_plan, t_data = st.tabs(
                ["📊 Chart", "🔍 Query plan", "📋 Data"])

            with t_chart:
                fig = make_chart(plan, df) if plan else None
                if fig:
                    st.plotly_chart(fig, width='stretch')
                elif df is not None and len(df) > 0:
                    st.caption("Single-value result — no chart needed.")
                else:
                    st.caption("No data to chart for this question.")

            with t_plan:
                meta = {
                    "Original question": result["question"],
                    "Routed by": result["routed_by"],
                    "Detected intent": route["intent"],
                    "Confidence": route["confidence"],
                    "Metrics used": ", ".join(route["metrics"]) or "—",
                    "Dimensions used": ", ".join(route["dimensions"]) or "—",
                    "Time window": plan["time_window"] if plan else "—",
                    "SQL template": plan["template_name"] if plan
                                    else "— (no SQL executed)",
                }
                if route["unsupported_reason"]:
                    meta["Refusal reason"] = route["unsupported_reason"]
                st.table(pd.DataFrame(meta.items(), columns=["Field", "Value"]))
                if plan:
                    st.code(plan["sql"], language="sql")
                else:
                    st.caption("Refused questions never reach SQL — that is "
                               "the governance working.")

            with t_data:
                if df is not None and len(df) > 0:
                    st.dataframe(df, width='stretch', hide_index=True)
                else:
                    st.caption("No rows — nothing was queried.")

# ---------------------------------------------------------------------------
# Page: Supported Questions
# ---------------------------------------------------------------------------
elif page == "❓ Supported Questions":
    st.subheader("Supported question families")
    st.write("Retail Assist deliberately supports a small set of question "
             "families and answers them correctly every time — instead of "
             "attempting everything and being unreliable.")
    try:
        qs = pd.read_csv(EVAL_DIR / "eval_questions.csv")
        sup = qs[qs["should_answer"] == "yes"][["question", "expected_intent"]]
        for intent, grp in sup.groupby("expected_intent"):
            st.markdown(f"**{intent.replace('_', ' ').title()}**")
            for q in grp["question"]:
                st.markdown(f"- {q}")
    except Exception:
        st.info("Eval question set not found.")

# ---------------------------------------------------------------------------
# Page: Semantic Layer
# ---------------------------------------------------------------------------
elif page == "📖 Semantic Layer":
    st.subheader("Governed semantic layer")
    st.write("Every metric the app can compute is defined once, here. "
             "AOV and margin rate are weighted ratios — computed at query "
             "time, never averaged across groups.")
    st.dataframe(pd.DataFrame(sem.definitions_table()),
                 width='stretch', hide_index=True)
    st.markdown("**Explicitly unsupported domains** (refused, not guessed):")
    st.markdown("\n".join(f"- {d}" for d in sem.unsupported_domains))

# ---------------------------------------------------------------------------
# Page: Evaluation Results
# ---------------------------------------------------------------------------
elif page == "✅ Evaluation Results":
    st.subheader("Evaluation results")
    st.write("Routing is scored against a labeled 50-question benchmark "
             "(35 supported across 7 families, 15 unsupported) — the same "
             "evaluation-first discipline used in production systems.")
    try:
        res = pd.read_csv(EVAL_DIR / "eval_results.csv")
        sup = res[res["should_answer"].str.lower() == "yes"]
        uns = res[res["should_answer"].str.lower() == "no"]
        c = st.columns(4)
        c[0].metric("Routing accuracy", f"{res['intent_ok'].mean():.0%}")
        c[1].metric("Metric selection", f"{sup['metrics_ok'].mean():.0%}")
        c[2].metric("Refusal accuracy", f"{uns['refusal_ok'].mean():.0%}")
        c[3].metric("Semantic correctness", f"{res['semantic_ok'].mean():.0%}")
        st.caption("Deterministic mode, point-in-time run. Re-runnable with "
                   "src/evaluator.py.")
        st.dataframe(res, width='stretch', hide_index=True)
    except Exception:
        st.info("Run src/evaluator.py to generate eval_results.csv.")

# ---------------------------------------------------------------------------
# Page: Architecture
# ---------------------------------------------------------------------------
elif page == "🏗️ Architecture":
    st.subheader("Architecture")
    st.markdown(f"""
This is a deliberately scaled-down public analogue. The production system I
led handled far broader scope; here the goal is a small supported set
answered correctly every time.

**Pipeline (every question):**
1. **Route** — deterministic keyword routing (or optional LLM intent parsing,
   validated against the same allowlists). Confidence below 0.4 → refusal.
2. **Validate** — metrics and dimensions checked against the governed
   semantic layer (`configs/metrics.yml`). Nothing outside it executes.
3. **Build SQL** — one of four static, pre-written templates. User text
   never enters SQL. No dynamic text-to-SQL, by design.
4. **Execute** — DuckDB over pre-aggregated parquet artifacts built offline
   by `data_prep.py`. The app never touches raw data at runtime.
5. **Answer** — deterministic business wording with enforced safety rules
   (no trend language on single rows, no causal language, nothing outside
   the dataframe). Optional LLM re-wording obeys the same rules and falls
   back silently on any failure.

**LLM role (when enabled):** `{GEMINI_MODEL}` parses intent into strict JSON
and words answers. It never generates SQL, never computes metrics, never
adds context. Any error — bad key, quota, retired model — falls back to
deterministic mode. The LLM is a language layer, not a dependency.
""")

# ---------------------------------------------------------------------------
# Page: Deliberate Limits
# ---------------------------------------------------------------------------
elif page == "⚠️ Deliberate Limits":
    st.warning(sem.data_info["disclosure"])
    st.subheader("Deliberate limits and non-goals")
    st.write("Strong systems have explicit boundaries. These are design "
             "decisions, not gaps.")
    limits = pd.DataFrame([
        ["Open SQL generation", "Out of scope",
         "All retrieval flows through static templates. No arbitrary querying."],
        ["Free-text metric computation", "Out of scope",
         "The model interprets retrieved outputs; it never computes business logic."],
        ["Forecasting / causality", "Out of scope",
         "Refused honestly rather than guessed."],
        ["Real-time data", "Out of scope",
         "Fixed public dataset (Dec 2010 – Dec 9, 2011). Reliability over recency."],
        ["Unmodeled domains", "Refused",
         "Unsupported coverage is a first-class response type."],
        ["LLM as dependency", "Avoided",
         "Deterministic mode is the default and the fallback. No key required."],
    ], columns=["Limit", "Status", "Rationale"])
    st.dataframe(limits, width='stretch', hide_index=True)
    st.caption("Note: this dataset is heavily wholesale, so AOV (~£534) is "
               "much higher than typical consumer retail — reported as-is.")
