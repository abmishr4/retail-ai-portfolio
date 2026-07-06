"""
Retail Rec — rerank.py

The business-rules layer: takes statistically scored candidates and applies
merchant-style constraints BEFORE anything reaches a customer. This
separation — learned/statistical relevance first, business rules second,
never mixed — is the core architectural principle the app demonstrates.

Rules applied (all knobs in configs/scoring_config.yml):
  1. Never recommend the anchor item itself.
  2. Hard-remove ineligible items (penalty >= 1.0 means remove).
  3. Hard-remove out-of-stock items (penalty >= 1.0 means remove).
  4. Same-category boost (small nudge toward coherent experiences).
  5. Low-confidence penalty below the confidence floor.
  6. Keep top final_top_k per anchor by final score.

Coverage discipline (fallback is a feature, not a bug):
  Anchors with fewer than final_top_k direct recommendations are filled
  from (a) same-category popular products, then (b) global popular
  products — in-stock, eligible, no duplicates, never the anchor — and
  every filled row is clearly labeled with its strategy.

Run (from repo root, after build_candidates.py):
    python projects/retail_rec/src/rerank.py
Writes artifacts/final_recommendations.parquet and artifacts/rerank_stats.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scoring import load_config, score_candidates
from reason_codes import assign_reason

BASE_DIR = Path(__file__).resolve().parent.parent          # projects/retail_rec
DEFAULT_ART_DIR = BASE_DIR / "artifacts"

FINAL_COLUMNS = [
    "anchor_stock_code", "anchor_description",
    "recommended_stock_code", "recommended_description",
    "rank", "final_score", "lift", "confidence", "co_purchase_count",
    "anchor_product_category", "recommended_product_category",
    "synthetic_in_stock", "synthetic_eligibility_flag",
    "strategy", "reason_code",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail Rec rerank + fallback")
    parser.add_argument("--art-dir", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    art_dir = Path(args.art_dir) if args.art_dir else DEFAULT_ART_DIR
    cfg = load_config(Path(args.config)) if args.config else load_config()
    rules = cfg["business_rules"]
    final_k = int(cfg["limits"]["final_top_k"])

    print("[1/5] Loading candidates and product lookup")
    cand = pd.read_parquet(art_dir / "co_purchase_candidates.parquet")
    lookup = pd.read_parquet(art_dir / "product_lookup.parquet")
    stats = {}

    # ------------------------------------------------------------------
    # 2. Statistical scoring (weights from YAML)
    # ------------------------------------------------------------------
    print("[2/5] Scoring candidates")
    cand = score_candidates(cand, cfg)

    # Join product attributes for both sides of every pair.
    a_cols = lookup[["stock_code", "description", "product_category"]].rename(
        columns={"stock_code": "anchor_stock_code",
                 "description": "anchor_description",
                 "product_category": "anchor_product_category"})
    r_cols = lookup[["stock_code", "description", "product_category",
                     "synthetic_in_stock", "synthetic_eligibility_flag"]].rename(
        columns={"stock_code": "recommended_stock_code",
                 "description": "recommended_description",
                 "product_category": "recommended_product_category"})
    cand = cand.merge(a_cols, on="anchor_stock_code", how="left")
    cand = cand.merge(r_cols, on="recommended_stock_code", how="left")

    # ------------------------------------------------------------------
    # 3. Business rules
    # ------------------------------------------------------------------
    print("[3/5] Applying business rules")
    before = len(cand)
    cand = cand[cand["anchor_stock_code"] != cand["recommended_stock_code"]]
    stats["self_pairs_removed"] = before - len(cand)

    if rules["ineligible_penalty"] >= 1.0:
        before = len(cand)
        cand = cand[cand["synthetic_eligibility_flag"]]
        stats["ineligible_removed"] = before - len(cand)
    if rules["out_of_stock_penalty"] >= 1.0:
        before = len(cand)
        cand = cand[cand["synthetic_in_stock"]]
        stats["out_of_stock_removed"] = before - len(cand)

    same_cat = (cand["anchor_product_category"]
                == cand["recommended_product_category"])
    low_conf = cand["confidence"] < float(rules["low_confidence_floor"])
    cand["final_score"] = (cand["base_score"]
                           + rules["same_category_boost"] * same_cat
                           - rules["low_confidence_penalty"] * low_conf
                           ).clip(lower=0).round(4)
    stats["same_category_boosted"] = int(same_cat.sum())
    stats["low_confidence_penalized"] = int(low_conf.sum())

    # Top-k direct recommendations per anchor.
    direct = (cand.sort_values(["anchor_stock_code", "final_score"],
                               ascending=[True, False])
                  .groupby("anchor_stock_code").head(final_k).copy())
    direct["strategy"] = "Direct co-purchase"

    # ------------------------------------------------------------------
    # 4. Fallback fill — category popularity, then global popularity
    # ------------------------------------------------------------------
    print("[4/5] Filling with fallbacks (coverage discipline)")
    sellable = lookup[lookup["synthetic_in_stock"]
                      & lookup["synthetic_eligibility_flag"]]
    global_pop = sellable.sort_values("order_count", ascending=False)
    cat_pop = {c: g.sort_values("order_count", ascending=False)
               for c, g in sellable.groupby("product_category")}

    anchor_info = lookup.set_index("stock_code")[
        ["description", "product_category"]]
    direct_by_anchor = {a: set(g["recommended_stock_code"])
                        for a, g in direct.groupby("anchor_stock_code")}

    fallback_rows = []
    for anchor, info in anchor_info.iterrows():
        have = direct_by_anchor.get(anchor, set())
        need = final_k - len(have)
        if need <= 0:
            continue
        exclude = have | {anchor}

        def take(pool, strategy, need, exclude):
            rows = []
            for _, p in pool.iterrows():
                if need <= 0:
                    break
                if p["stock_code"] in exclude:
                    continue
                rows.append({
                    "anchor_stock_code": anchor,
                    "anchor_description": info["description"],
                    "recommended_stock_code": p["stock_code"],
                    "recommended_description": p["description"],
                    "final_score": np.nan, "lift": np.nan,
                    "confidence": np.nan, "co_purchase_count": np.nan,
                    "anchor_product_category": info["product_category"],
                    "recommended_product_category": p["product_category"],
                    "synthetic_in_stock": True,
                    "synthetic_eligibility_flag": True,
                    "strategy": strategy,
                })
                exclude.add(p["stock_code"])
                need -= 1
            return rows, need, exclude

        pool = cat_pop.get(info["product_category"], global_pop).head(60)
        rows, need, exclude = take(pool, "Category popularity fallback",
                                   need, exclude)
        fallback_rows.extend(rows)
        if need > 0:
            rows, need, exclude = take(global_pop.head(80),
                                       "Global popularity fallback",
                                       need, exclude)
            fallback_rows.extend(rows)

    fallback = pd.DataFrame(fallback_rows)
    stats["fallback_rows_added"] = len(fallback)

    # ------------------------------------------------------------------
    # 5. Assemble final artifact: direct first, fallbacks after
    # ------------------------------------------------------------------
    print("[5/5] Writing final_recommendations.parquet")
    final = pd.concat([direct, fallback], ignore_index=True)
    final["_direct"] = (final["strategy"] == "Direct co-purchase").astype(int)
    final = final.sort_values(
        ["anchor_stock_code", "_direct", "final_score"],
        ascending=[True, False, False])
    final["rank"] = final.groupby("anchor_stock_code").cumcount() + 1
    final = final[final["rank"] <= final_k]

    final["reason_code"] = final.apply(
        lambda r: assign_reason(
            r["strategy"], r["co_purchase_count"], r["confidence"],
            r["lift"],
            r["anchor_product_category"] == r["recommended_product_category"]),
        axis=1)

    final = final[FINAL_COLUMNS].reset_index(drop=True)
    final.to_parquet(art_dir / "final_recommendations.parquet",
                     compression="snappy", index=False)

    stats["anchors_covered"] = int(final["anchor_stock_code"].nunique())
    stats["total_rows"] = len(final)
    stats["direct_share"] = round(
        float((final["strategy"] == "Direct co-purchase").mean()), 4)
    with open(art_dir / "rerank_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    for k, v in stats.items():
        print(f"      {k}: {v}")
    print(f"      Output: {art_dir / 'final_recommendations.parquet'}")


if __name__ == "__main__":
    main()
