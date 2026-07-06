"""
Retail Rec — evaluate.py

Offline evaluation ONLY. This deliberately does not claim online lift,
A/B-test results, or revenue impact — the UCI dataset cannot support such
claims and this demo does not fake them. What offline evaluation CAN
honestly measure is coverage, evidence quality, and rule activity:

  * coverage: every product should have a full recommendation list
  * evidence: how strong the direct co-purchase signal is (lift, confidence)
  * fallback discipline: how much of the surface relies on fallbacks
  * rule activity: what the business-rules layer actually removed

Run (from repo root, after rerank.py):
    python projects/retail_rec/src/evaluate.py
Writes artifacts/metrics.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent          # projects/retail_rec
DEFAULT_ART_DIR = BASE_DIR / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail Rec offline evaluation")
    parser.add_argument("--art-dir", default=None)
    args = parser.parse_args()
    art_dir = Path(args.art_dir) if args.art_dir else DEFAULT_ART_DIR

    final = pd.read_parquet(art_dir / "final_recommendations.parquet")
    lookup = pd.read_parquet(art_dir / "product_lookup.parquet")
    with open(art_dir / "rerank_stats.json", "r", encoding="utf-8") as f:
        rerank_stats = json.load(f)

    n_products = lookup["stock_code"].nunique()
    per_anchor = final.groupby("anchor_stock_code")
    direct = final[final["strategy"] == "Direct co-purchase"]
    direct_per_anchor = direct.groupby("anchor_stock_code").size()

    metrics = {
        "catalog_products": int(n_products),
        "products_with_at_least_1_rec": int(final["anchor_stock_code"].nunique()),
        "products_with_full_20_recs": int((per_anchor.size() == 20).sum()),
        "avg_direct_recs_per_product": round(
            float(direct_per_anchor.reindex(lookup["stock_code"])
                  .fillna(0).mean()), 2),
        "fallback_rate_rows": round(
            float((final["strategy"] != "Direct co-purchase").mean()), 4),
        "products_using_any_fallback": int(
            final[final["strategy"] != "Direct co-purchase"]
            ["anchor_stock_code"].nunique()),
        "avg_lift_direct": round(float(direct["lift"].mean()), 2),
        "median_lift_direct": round(float(direct["lift"].median()), 2),
        "avg_confidence_direct": round(float(direct["confidence"].mean()), 4),
        "category_diversity_avg_unique_categories_per_anchor": round(
            float(per_anchor["recommended_product_category"]
                  .nunique().mean()), 2),
        "out_of_stock_removed": int(rerank_stats.get("out_of_stock_removed", 0)),
        "ineligible_removed": int(rerank_stats.get("ineligible_removed", 0)),
        "low_confidence_penalized": int(
            rerank_stats.get("low_confidence_penalized", 0)),
        "same_category_boosted": int(
            rerank_stats.get("same_category_boosted", 0)),
        "note": ("Offline metrics only. No online lift, A/B-test, or "
                 "revenue claims are made or supported by this demo."),
    }

    out_path = art_dir / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Retail Rec — offline evaluation")
    print("-" * 44)
    for k, v in metrics.items():
        if k != "note":
            print(f"{k:52s} {v}")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
