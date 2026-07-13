"""
Retail Rec — app.py

Public portfolio demo: a "Customers Also Bought" recommendation experience
built on transparent co-purchase statistics and visible business rules.

The app performs NO computation at startup — it only reads pre-built
artifacts (product_lookup.parquet, final_recommendations.parquet,
metrics.json). All mining, scoring, reranking, and fallback logic runs
offline in the src/ pipeline.
"""

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from product_search import search_products, NO_RESULTS_HINT   # noqa: E402

ART_DIR = BASE_DIR / "artifacts"
CONFIG_PATH = BASE_DIR / "configs" / "scoring_config.yml"

ACCENT = "#6366F1"

DISCLOSURE = ("This demo uses the UCI Online Retail Dataset (public) for "
              "transaction data. Category, inventory, margin, and "
              "eligibility fields are synthetic enrichments. No employer "
              "data, employer code, or confidential business logic is "
              "included.")

st.set_page_config(page_title="Retail Rec", page_icon="🛍️", layout="wide")


# ---------------------------------------------------------------------------
# Styling — a light CSS pass so the recommendation surface reads as a 2026
# product experience, not a raw table. Theme-neutral (rgba / inherit) so it
# adapts to light and dark.
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container {padding-top: 2.2rem; max-width: 1100px;}

          /* recommendation card */
          .rr-card {border:1px solid rgba(128,128,128,.22); border-radius:14px;
                    padding:.85rem 1rem; margin:.55rem 0;
                    background:rgba(99,102,241,.035);}
          .rr-card.fallback {background:rgba(245,158,11,.05);
                             border-color:rgba(245,158,11,.28);
                             border-style:dashed;}
          .rr-head {display:flex; align-items:center; gap:.55rem;
                    flex-wrap:wrap;}
          .rr-rank {font-size:.8rem; font-weight:700; opacity:.55;
                    min-width:1.9rem;}
          .rr-title {font-weight:650; font-size:1.02rem; flex:1;}
          .rr-badge {font-size:.68rem; font-weight:700; letter-spacing:.02em;
                     padding:.16rem .55rem; border-radius:999px;
                     text-transform:uppercase;}
          .rr-badge.direct {background:rgba(99,102,241,.16); color:inherit;
                            border:1px solid rgba(99,102,241,.35);}
          .rr-badge.fallback {background:rgba(245,158,11,.16); color:inherit;
                              border:1px solid rgba(245,158,11,.4);}

          .rr-chips {display:flex; flex-wrap:wrap; gap:.4rem;
                     margin:.5rem 0 .35rem;}
          .rr-chip {font-size:.72rem; font-weight:500; line-height:1.2;
                    padding:.18rem .58rem; border-radius:999px; color:inherit;
                    background:rgba(99,102,241,.1);
                    border:1px solid rgba(99,102,241,.24); white-space:nowrap;}
          .rr-chip b {font-weight:700;}
          .rr-chip.cat {background:rgba(128,128,128,.12);
                        border-color:rgba(128,128,128,.28);}

          /* score bar */
          .rr-bar {height:6px; border-radius:999px;
                   background:rgba(128,128,128,.16); overflow:hidden;
                   margin:.15rem 0 .45rem;}
          .rr-bar > div {height:100%; border-radius:999px;
                         background:linear-gradient(90deg,#6366F1,#8B5CF6);}
          .rr-reason {font-size:.82rem; opacity:.75; font-style:italic;}

          .stButton>button {border-radius:12px;
                            border:1px solid rgba(128,128,128,.28);
                            font-weight:500;}
          .stButton>button:hover {border-color:rgba(99,102,241,.6);}

          .rr-hero-title {font-size:1.55rem; font-weight:700; margin:.2rem 0 .1rem;}
          .rr-hero-sub {opacity:.75; font-size:.95rem; margin-bottom:.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached artifact loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_artifacts():
    lookup = pd.read_parquet(ART_DIR / "product_lookup.parquet")
    recs = pd.read_parquet(ART_DIR / "final_recommendations.parquet")
    with open(ART_DIR / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return lookup, recs, metrics, cfg


# ---------------------------------------------------------------------------
# Recommendation cards (the customer-facing surface)
# ---------------------------------------------------------------------------
def _rec_card_html(row, max_score: float) -> str:
    direct = row.strategy == "Direct co-purchase"
    title = html.escape(str(row.recommended_description).title())
    category = html.escape(str(row.recommended_product_category))
    reason = html.escape(str(row.reason_code))

    if direct:
        badge = "<span class='rr-badge direct'>Direct co-purchase</span>"
        chips = [f"<span class='rr-chip cat'>{category}</span>"]
        if pd.notna(row.lift):
            chips.append(f"<span class='rr-chip'>lift · <b>{row.lift:.1f}</b></span>")
        if pd.notna(row.confidence):
            chips.append(f"<span class='rr-chip'>confidence · <b>{row.confidence:.2f}</b></span>")
        if pd.notna(row.co_purchase_count):
            chips.append(f"<span class='rr-chip'>baskets · <b>{int(row.co_purchase_count)}</b></span>")
        if pd.notna(row.final_score):
            chips.append(f"<span class='rr-chip'>score · <b>{row.final_score:.3f}</b></span>")
        pct = 0 if not max_score or pd.isna(row.final_score) else \
            max(6, min(100, row.final_score / max_score * 100))
        bar = f"<div class='rr-bar'><div style='width:{pct:.0f}%'></div></div>"
        card_cls = "rr-card"
    else:
        badge = "<span class='rr-badge fallback'>Popularity fallback</span>"
        chips = [f"<span class='rr-chip cat'>{category}</span>"]
        bar = ""
        card_cls = "rr-card fallback"

    chips_html = "".join(chips)
    return (
        f"<div class='{card_cls}'>"
        f"<div class='rr-head'><span class='rr-rank'>#{int(row.rank)}</span>"
        f"<span class='rr-title'>{title}</span>{badge}</div>"
        f"<div class='rr-chips'>{chips_html}</div>"
        f"{bar}"
        f"<div class='rr-reason'>{reason}</div>"
        f"</div>"
    )


def render_recommendations(r: pd.DataFrame) -> None:
    direct = r[r["strategy"] == "Direct co-purchase"]
    max_score = float(direct["final_score"].max()) if len(direct) else 0.0
    cards = "".join(_rec_card_html(row, max_score) for row in r.itertuples())
    st.markdown(cards, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# App body
# ---------------------------------------------------------------------------
inject_css()
lookup, recs, metrics, cfg = load_artifacts()

if "selected_code" not in st.session_state:
    st.session_state.selected_code = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None

# ---------------------------------------------------------------------------
# Sidebar — navigation + disclosure
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛍️ Retail Rec")
    st.caption("'Customers Also Bought' on transparent co-purchase "
               "statistics — public demo")

    page = st.radio(
        "Navigate",
        ["🛒 Shop & Recommendations", "🧮 Recommendation Logic",
         "📈 Coverage & Metrics", "⚠️ Deliberate Limits"],
        label_visibility="collapsed")

    st.divider()
    st.caption(DISCLOSURE)
    st.caption("Companion app: Retail Assist — governed natural-language "
               "retail analytics (see repo README for link).")

# ---------------------------------------------------------------------------
# Page: Shop & Recommendations (search -> select -> recs, one page)
# ---------------------------------------------------------------------------
if page == "🛒 Shop & Recommendations":
    st.markdown(
        "<div class='rr-hero-title'>🛍️ Customers Also Bought</div>"
        "<div class='rr-hero-sub'>Pick a product from a real 3,922-item "
        "catalog and see its recommendations — each with its co-purchase "
        "evidence, score, strategy, and an honest reason. Statistical "
        "relevance and business rules are kept in separate layers.</div>",
        unsafe_allow_html=True)

    # Trending products — one-click entry points so visitors don't face a
    # blank search box over an unfamiliar 3,922-product catalog.
    # "Trending" = most-ordered in-stock products (deterministic, honest).
    trending = (lookup[lookup["synthetic_in_stock"]]
                .nlargest(8, "order_count"))
    st.caption("🔥 Trending products — click to see what customers also bought")
    rows = [trending.iloc[:4], trending.iloc[4:]]
    for chunk in rows:
        cols = st.columns(4)
        for col, p in zip(cols, chunk.itertuples()):
            label = p.description.title()
            label = label if len(label) <= 32 else label[:29] + "..."
            if col.button(label, key=f"trend_{p.stock_code}", width='stretch',
                          help=f"{p.description} · {int(p.order_count):,} orders"):
                st.session_state.selected_code = p.stock_code
                st.session_state.search_results = None

    with st.form("search_form"):
        c1, c2 = st.columns([5, 1])
        query = c1.text_input("Search by name or stock code",
                              placeholder="e.g., regency teacup · jumbo bag · 22699",
                              label_visibility="collapsed")
        submitted = c2.form_submit_button("Search", type="primary",
                                          width='stretch')
    if submitted:
        st.session_state.search_results = search_products(lookup, query)
        st.session_state.selected_code = None

    results = st.session_state.search_results
    if results is not None and len(results) == 0:
        st.warning(NO_RESULTS_HINT)

    if results is not None and len(results) > 0:
        options = {f"{r.description}  ·  {r.stock_code}": r.stock_code
                   for r in results.itertuples()}
        choice = st.selectbox(f"{len(results)} matches — pick a product",
                              list(options.keys()))
        st.session_state.selected_code = options[choice]

    code = st.session_state.selected_code
    if code is None:
        st.info("👋 **Welcome to Retail Rec.** Search the catalog above "
                "(3,922 products from a real UK online retailer), pick a "
                "product, and see its 'Customers Also Bought' "
                "recommendations — each with the co-purchase evidence, "
                "score, strategy, and a plain-language reason. Try "
                "**regency teacup** or **jumbo bag** for strong examples, "
                "or an obscure product to see fallback handling.")
    else:
        # ----- Selected product card -----
        p = lookup[lookup["stock_code"] == code].iloc[0]
        st.divider()
        st.markdown(f"### {p.description.title()}")
        c = st.columns(6)
        c[0].metric("Stock code", p.stock_code)
        c[1].metric("Category*", p.product_category)
        c[2].metric("Price tier*", p.synthetic_price_bucket)
        c[3].metric("Orders", f"{int(p.order_count):,}")
        c[4].metric("Popularity rank", f"#{int(p.product_popularity_rank):,}")
        c[5].metric("In stock*", "Yes" if p.synthetic_in_stock else "No")
        st.caption("*synthetic demo fields")

        # ----- Recommendations -----
        r = recs[recs["anchor_stock_code"] == code].sort_values("rank")
        direct_n = int((r["strategy"] == "Direct co-purchase").sum())
        fallback_n = len(r) - direct_n

        st.markdown("### Customers Also Bought")
        m = st.columns(3)
        m[0].metric("Recommendations", len(r))
        m[1].metric("Direct co-purchase", direct_n)
        m[2].metric("Labeled fallbacks", fallback_n)
        st.caption("Direct rows carry co-purchase evidence; fallback rows are "
                   "clearly-labeled popularity fills — never disguised as "
                   "co-purchase evidence.")

        render_recommendations(r)

        # Browse: jump to any recommended product's own recommendations.
        # Keying on `code` gives a fresh (reset) selectbox per anchor, so
        # selecting one navigates exactly once with no rerun loop.
        rec_map = {f"{row.recommended_description.title()}  ·  {row.recommended_stock_code}":
                   row.recommended_stock_code for row in r.itertuples()}
        pick = st.selectbox(
            "🔎 Explore a recommended product's own recommendations",
            ["—"] + list(rec_map.keys()), key=f"explore_{code}")
        if pick != "—":
            st.session_state.selected_code = rec_map[pick]
            st.session_state.search_results = None
            st.rerun()

        with st.expander("🔬 Full table (every column and evidence value)"):
            show = r[["rank", "recommended_description",
                      "recommended_product_category", "final_score", "lift",
                      "confidence", "co_purchase_count", "strategy",
                      "reason_code"]].rename(columns={
                          "recommended_description": "product",
                          "recommended_product_category": "category"})
            st.dataframe(
                show, width='stretch', hide_index=True, height=560,
                column_config={
                    "rank": st.column_config.NumberColumn("#", width="small"),
                    "final_score": st.column_config.NumberColumn(
                        "score", format="%.3f",
                        help="YAML-weighted blend of lift, confidence, and "
                             "log co-purchase count, after business rules"),
                    "lift": st.column_config.NumberColumn(
                        "lift", format="%.1f",
                        help="How much more likely this pairing is than the "
                             "item's baseline popularity. Read alongside the "
                             "co-purchase count — lift on 1–2 baskets is noisy."),
                    "confidence": st.column_config.NumberColumn(
                        "confidence", format="%.2f",
                        help="P(this item in basket | anchor in basket)"),
                    "co_purchase_count": st.column_config.NumberColumn(
                        "baskets", help="Baskets containing both items"),
                })

# ---------------------------------------------------------------------------
# Page: Recommendation Logic
# ---------------------------------------------------------------------------
elif page == "🧮 Recommendation Logic":
    st.subheader("How recommendations are produced")

    st.markdown("""
**1 · Candidate generation (statistics, no model server needed)**

Every pair of products bought in the same basket is counted across 16,217
mined baskets (single-item baskets skipped; baskets over 50 unique items
skipped as wholesale noise). Three statistics per directional pair:

- **co-purchase count** — baskets containing both items (the evidence volume)
- **confidence** — `P(recommended in basket | anchor in basket)`
- **lift** — confidence ÷ the recommended item's baseline basket share.
  Lift > 1 means the pairing beats chance; lift 18 means an 18× stronger
  association than popularity alone would predict. **Lift on 1–2 baskets is
  statistically noisy** — which is why evidence volume is scored too, and
  why the UI always shows the basket count next to the lift.
""")

    st.markdown("**2 · Scoring (weights live in YAML, not code)**")
    w = cfg["retrieval_weights"]
    st.code(
        f"base_score = {w['lift']:.2f} × normalized(log lift)\n"
        f"           + {w['confidence']:.2f} × normalized(confidence)\n"
        f"           + {w['co_purchase_count']:.2f} × normalized(log co-purchase count)",
        language="text")
    st.caption("Lift and count are log-scaled before normalization because "
               "both are heavy-tailed — raw min-max would let a handful of "
               "extreme values flatten every real signal.")

    st.markdown("**3 · Business rules (applied after statistics, never mixed in)**")
    b = cfg["business_rules"]
    rules = pd.DataFrame([
        ["Out-of-stock items", "Removed entirely",
         f"penalty = {b['out_of_stock_penalty']:.2f} (≥1.0 ⇒ hard filter)"],
        ["Ineligible items", "Removed entirely",
         f"penalty = {b['ineligible_penalty']:.2f} (≥1.0 ⇒ hard filter)"],
        ["Same-category pairings", f"Boosted +{b['same_category_boost']:.2f}",
         "small nudge toward coherent shopping experiences"],
        [f"Confidence below {b['low_confidence_floor']:.2f}",
         f"Penalized −{b['low_confidence_penalty']:.2f}",
         "thin conditional probability ranks lower"],
        ["Near-duplicate recommendations",
         f"Diversified (MMR, weight {cfg.get('diversity', {}).get('mmr_weight', 0):.2f})",
         "each pick is penalized by its name-token overlap with items "
         "already selected — dampens 'five cakestands' without killing "
         "legitimate colorway variants. Display order follows the "
         "diversified selection, so scores are intentionally "
         "non-monotonic."],
    ], columns=["Rule", "Effect", "Detail"])
    st.dataframe(rules, width='stretch', hide_index=True)

    st.info("**A real example from this dataset:** for ROSES REGENCY TEACUP "
            "AND SAUCER, the statistically #2 recommendation (PINK REGENCY "
            "TEACUP, 350 shared baskets) is absent from the final list — "
            "its synthetic inventory flag is out-of-stock, so the business-"
            "rules layer removed it before serving. Statistical relevance "
            "and business constraints are kept in strictly separate "
            "layers: rules can change without touching the statistics.")

    st.markdown("**4 · Fallbacks (coverage discipline)**")
    st.markdown("""
Products with fewer than 20 direct recommendations are filled from
same-category popular products, then global popular products — in-stock,
eligible, never the anchor, never a duplicate. Every fill is labeled with
its strategy and an honest reason code. In this dataset, category pools
always sufficed, so the global-popularity path exists but never fires —
reported as-is.

**5 · Reason codes (the customer-value layer)** — every row carries a
plain-language reason graded by actual evidence, from *"Frequently bought
together in the same basket"* down to *"Co-purchased in a small number of
baskets — limited evidence, ranked accordingly."* Thin evidence says so.
""")

# ---------------------------------------------------------------------------
# Page: Coverage & Metrics
# ---------------------------------------------------------------------------
elif page == "📈 Coverage & Metrics":
    st.subheader("Offline evaluation")
    st.write("Coverage and evidence quality, measured offline. No online "
             "lift, A/B-test, or revenue claims are made — this dataset "
             "cannot support them and this demo does not fake them.")

    c = st.columns(4)
    c[0].metric("Products covered",
                f"{metrics['products_with_at_least_1_rec']:,} / "
                f"{metrics['catalog_products']:,}")
    c[1].metric("Full 20-rec lists",
                f"{metrics['products_with_full_20_recs']:,}")
    c[2].metric("Direct recs / product",
                f"{metrics['avg_direct_recs_per_product']:.1f}")
    c[3].metric("Fallback rate", f"{metrics['fallback_rate_rows']:.1%}")

    c = st.columns(4)
    c[0].metric("Median lift (direct)", f"{metrics['median_lift_direct']:.1f}",
                help="Median, not mean — mean lift is inflated by "
                     "single-basket pairs with extreme values "
                     f"(mean here: {metrics['avg_lift_direct']:.0f}).")
    c[1].metric("Avg confidence (direct)",
                f"{metrics['avg_confidence_direct']:.2f}")
    c[2].metric("Category diversity",
                f"{metrics['category_diversity_avg_unique_categories_per_anchor']:.1f}",
                help="Average unique recommended categories per product")
    c[3].metric("Products using fallback",
                f"{metrics['products_using_any_fallback']:,}")

    st.markdown("**Business-rules activity** (what the rerank layer did)")
    c = st.columns(4)
    c[0].metric("Out-of-stock removed", f"{metrics['out_of_stock_removed']:,}")
    c[1].metric("Ineligible removed", f"{metrics['ineligible_removed']:,}")
    c[2].metric("Low-confidence penalized",
                f"{metrics['low_confidence_penalized']:,}")
    c[3].metric("Same-category boosted",
                f"{metrics['same_category_boosted']:,}")
    st.caption("Regenerate anytime: python src/evaluate.py")

# ---------------------------------------------------------------------------
# Page: Deliberate Limits
# ---------------------------------------------------------------------------
elif page == "⚠️ Deliberate Limits":
    st.warning(DISCLOSURE)
    st.subheader("Deliberate limits and non-goals")
    st.write("Strong systems have explicit boundaries. These are design "
             "decisions, not gaps.")
    limits = pd.DataFrame([
        ["Online lift / A/B claims", "None made",
         "This is offline co-purchase logic, not a live experiment. No "
         "UPT or revenue lift is claimed."],
        ["Deep learning retrieval", "Out of scope for this MVP",
         "Co-purchase + lift + confidence + reranking demonstrates the "
         "system thinking without a model server."],
        ["Employer ranking logic", "Not used",
         "Public-data analogue only; no proprietary rules or schemas."],
        ["Dataset scale", "Small and sparse",
         "UCI Online Retail (~20K baskets, one UK wholesaler) is tiny "
         "versus enterprise commerce data — fallback behavior reflects "
         "that honestly."],
        ["Inventory & eligibility", "Synthetic",
         "Deterministic demo fields to exercise the business-rules layer; "
         "not real stock states."],
        ["Category labels", "Synthetic, keyword-based",
         "Occasional misfires (a 'fairy cake umbrella' lands in Kitchen & "
         "Dining) — an accepted artifact of demo enrichment."],
        ["Personalization", "Out of scope",
         "Recommendations are anchor-product-based, not user-based — no "
         "customer profiles or session signals in this demo."],
    ], columns=["Limit", "Status", "Rationale"])
    st.dataframe(limits, width='stretch', hide_index=True)
