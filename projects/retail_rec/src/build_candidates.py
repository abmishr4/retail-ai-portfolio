"""
Retail Rec — build_candidates.py

Candidate generation: turns basket-level transactions into directional
co-purchase candidates (anchor -> recommended) with the three statistics the
whole system runs on:

  confidence = co_purchase_count / anchor_basket_count
               P(recommended in basket | anchor in basket)
  support    = co_purchase_count / total_baskets
  lift       = confidence / (recommended_basket_count / total_baskets)
               how much MORE likely the pairing is than the recommended
               item's baseline popularity (lift > 1 = real signal)

Memory discipline (deliberate — this is the file that would blow up naively):
  * NO full item-item matrix, NO crosstab, NO cross-join, NO pivot.
  * Per-basket itertools.combinations over unique items, aggregated with a
    Counter — linear in actual co-occurrences, not catalog^2.
  * Baskets larger than max_large_basket_size are skipped as wholesale
    noise (a 200-item basket generates ~20k near-meaningless pairs).
  * Only the top intermediate_top_k candidates per anchor are saved.

Run (from repo root, after data_prep.py):
    python projects/retail_rec/src/build_candidates.py
"""

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent          # projects/retail_rec
DEFAULT_DATA_DIR = BASE_DIR / "data" / "processed"
DEFAULT_ART_DIR = BASE_DIR / "artifacts"
DEFAULT_CONFIG = BASE_DIR / "configs" / "scoring_config.yml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail Rec candidate generation")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--art-dir", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    art_dir = Path(args.art_dir) if args.art_dir else DEFAULT_ART_DIR
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    art_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    max_basket = int(cfg["limits"]["max_large_basket_size"])
    top_k = int(cfg["limits"]["intermediate_top_k"])
    min_co = int(cfg["limits"]["min_co_purchase_count"])

    # ------------------------------------------------------------------
    # 1. Baskets -> unique item sets
    # ------------------------------------------------------------------
    print("[1/4] Reading cleaned order lines and forming baskets")
    lines = pd.read_parquet(data_dir / "cleaned_order_lines.parquet",
                            columns=["basket_id", "stock_code"])
    baskets = lines.groupby("basket_id")["stock_code"].apply(
        lambda s: sorted(set(s)))

    total_baskets_all = len(baskets)
    sizes = baskets.map(len)
    skipped_large = int((sizes > max_basket).sum())
    skipped_single = int((sizes < 2).sum())
    baskets = baskets[(sizes >= 2) & (sizes <= max_basket)]
    total_baskets = len(baskets)   # denominator: baskets actually mined
    print(f"      Baskets total: {total_baskets_all:,} | single-item skipped: "
          f"{skipped_single:,} | >{max_basket}-item skipped: {skipped_large:,} "
          f"| mined: {total_baskets:,}")

    # ------------------------------------------------------------------
    # 2. Pair counting — Counter over per-basket combinations
    # ------------------------------------------------------------------
    print("[2/4] Counting co-purchase pairs (itertools.combinations)")
    pair_counts: Counter = Counter()
    item_basket_counts: Counter = Counter()
    for items in baskets:
        item_basket_counts.update(items)
        pair_counts.update(combinations(items, 2))   # items pre-sorted
    print(f"      Unique unordered pairs: {len(pair_counts):,}")

    # ------------------------------------------------------------------
    # 3. Directional candidates with confidence / support / lift
    # ------------------------------------------------------------------
    print("[3/4] Computing confidence, support, lift (directional)")
    pairs = pd.DataFrame(
        [(a, b, c) for (a, b), c in pair_counts.items()],
        columns=["item_a", "item_b", "co_purchase_count"])
    pairs = pairs[pairs["co_purchase_count"] >= min_co]

    # Each unordered pair becomes two directional rows: a->b and b->a.
    fwd = pairs.rename(columns={"item_a": "anchor_stock_code",
                                "item_b": "recommended_stock_code"})
    bwd = pairs.rename(columns={"item_b": "anchor_stock_code",
                                "item_a": "recommended_stock_code"})
    cand = pd.concat([fwd, bwd], ignore_index=True)

    ibc = pd.Series(item_basket_counts, name="cnt")
    cand["anchor_basket_count"] = cand["anchor_stock_code"].map(ibc)
    cand["recommended_basket_count"] = cand["recommended_stock_code"].map(ibc)
    cand["total_baskets"] = total_baskets

    cand["confidence"] = cand["co_purchase_count"] / cand["anchor_basket_count"]
    cand["support"] = cand["co_purchase_count"] / total_baskets
    expected_conf = cand["recommended_basket_count"] / total_baskets
    cand["lift"] = cand["confidence"] / expected_conf

    # Keep only the strongest top_k candidates per anchor (by evidence,
    # then lift) — the full pair set never needs to leave this script.
    cand = (cand.sort_values(["anchor_stock_code", "co_purchase_count", "lift"],
                             ascending=[True, False, False])
                .groupby("anchor_stock_code")
                .head(top_k)
                .reset_index(drop=True))

    for col in ["confidence", "support", "lift"]:
        cand[col] = cand[col].astype(np.float32)

    out_path = art_dir / "co_purchase_candidates.parquet"
    cand.to_parquet(out_path, compression="snappy", index=False)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("[4/4] Done. Summary:")
    print(f"      Directional candidate rows: {len(cand):,}")
    print(f"      Anchors with >=1 candidate: {cand['anchor_stock_code'].nunique():,}")
    print(f"      Median candidates/anchor:   {cand.groupby('anchor_stock_code').size().median():.0f}")
    print(f"      Lift  p50/p90:              {cand['lift'].quantile(0.5):.1f} / {cand['lift'].quantile(0.9):.1f}")
    print(f"      Confidence p50/p90:         {cand['confidence'].quantile(0.5):.3f} / {cand['confidence'].quantile(0.9):.3f}")
    print(f"      Output: {out_path}")


if __name__ == "__main__":
    main()
