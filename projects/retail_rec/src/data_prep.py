"""
Retail Rec — data_prep.py

One-time offline preparation for the recommendation pipeline. Converts the
raw UCI Online Retail Excel file into:
  * cleaned_order_lines.parquet  (basket-level transaction lines)
  * product_lookup.parquet       (one row per product, with popularity and
                                  synthetic demo fields)

The Streamlit app NEVER reads Excel or raw data — it reads only the small
artifacts produced by this pipeline.

Run (from repo root):
    python projects/retail_rec/src/data_prep.py
Or point at any file explicitly:
    python data_prep.py --raw "/path/to/Online Retail.xlsx" --out-data X --out-art Y

Synthetic enrichment disclosure:
    product_category, synthetic_price_bucket, synthetic_margin_rate,
    synthetic_in_stock, and synthetic_eligibility_flag are SYNTHETIC demo
    fields, deterministically generated (stable-hash / rule based, no
    randomness). They do not represent any real retailer's business logic.
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent          # projects/retail_rec
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_DATA_DIR = BASE_DIR / "data" / "processed"
DEFAULT_ART_DIR = BASE_DIR / "artifacts"

# ---------------------------------------------------------------------------
# Synthetic enrichment rules — identical keyword map to Retail Assist so the
# two portfolio apps tell one consistent story about the same catalog.
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = [
    ("Christmas & Seasonal", ["CHRISTMAS", "XMAS", "SANTA", "REINDEER", "ADVENT", "EASTER", "HALLOWEEN"]),
    ("Candles & Fragrance",  ["CANDLE", "T-LIGHT", "TEALIGHT", "INCENSE", "SCENT", "FRAGRANCE"]),
    ("Kitchen & Dining",     ["MUG", "CUP", "TEACUP", "TEAPOT", "BOWL", "PLATE", "CUTLERY", "JUG",
                              "CAKESTAND", "CAKE", "BAKING", "JAR", "TIN", "TRAY", "APRON", "NAPKIN",
                              "COASTER", "SPOON", "KITCHEN", "LUNCH BOX", "EGG"]),
    ("Bags & Accessories",   ["BAG", "PURSE", "WALLET", "UMBRELLA", "SCARF", "HANDBAG", "JEWELLERY",
                              "NECKLACE", "BRACELET", "EARRING", "RING ", "HAIR"]),
    ("Stationery & Crafts",  ["CARD", "PEN", "PENCIL", "NOTEBOOK", "PAPER", "TAPE", "STICKER",
                              "CHALK", "CRAYON", "ENVELOPE", "GIFT WRAP", "WRAP", "RIBBON", "CRAFT"]),
    ("Toys & Games",         ["TOY", "GAME", "PUZZLE", "DOLL", "SOLDIER", "SPACEBOY", "PLAYHOUSE",
                              "BUILDING BLOCK", "SKITTLES", "DOMINO", "KIDS", "CHILDREN"]),
    ("Garden & Outdoor",     ["GARDEN", "PLANT", "WATERING", "PARASOL", "BIRD", "LANTERN",
                              "WINDMILL", "OUTDOOR", "PICNIC"]),
    ("Home Decor",           ["HEART", "SIGN", "FRAME", "CUSHION", "DOORMAT", "HOOK", "DRAWER",
                              "SHELF", "MIRROR", "CLOCK", "ORNAMENT", "DECORATION", "BUNTING",
                              "WICKER", "LIGHT", "HANGING", "HOLDER", "BOX"]),
]
DEFAULT_CATEGORY = "General Merchandise"

CATEGORY_MARGIN_RATE = {
    "Christmas & Seasonal": 0.52, "Candles & Fragrance": 0.55,
    "Kitchen & Dining": 0.48, "Bags & Accessories": 0.50,
    "Stationery & Crafts": 0.58, "Toys & Games": 0.45,
    "Garden & Outdoor": 0.42, "Home Decor": 0.50,
    DEFAULT_CATEGORY: 0.46,
}


def stable_bucket(key: str, buckets: int) -> int:
    """Deterministic 0..buckets-1 assignment from a string key (md5-based,
    stable across runs/machines — no RNG, fully reproducible)."""
    return int(hashlib.md5(str(key).encode("utf-8")).hexdigest(), 16) % buckets


def assign_category(description: str) -> str:
    d = str(description).upper()
    for category, keywords in CATEGORY_KEYWORDS:
        if any(k in d for k in keywords):
            return category
    return DEFAULT_CATEGORY


def find_raw_file(raw_arg: str | None) -> Path:
    if raw_arg:
        p = Path(raw_arg)
        if not p.exists():
            raise FileNotFoundError(f"--raw path not found: {p}")
        return p
    candidates = sorted(DEFAULT_RAW_DIR.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(
            f"No .xlsx found in {DEFAULT_RAW_DIR}. Download the UCI Online "
            "Retail dataset there, or pass --raw /path/to/file.xlsx")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail Rec data prep")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--out-data", default=None)
    parser.add_argument("--out-art", default=None)
    args = parser.parse_args()

    raw_path = find_raw_file(args.raw)
    data_dir = Path(args.out_data) if args.out_data else DEFAULT_DATA_DIR
    art_dir = Path(args.out_art) if args.out_art else DEFAULT_ART_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raw Excel -> parquet immediately; all later reads = parquet
    # ------------------------------------------------------------------
    print(f"[1/4] Loading raw Excel: {raw_path}")
    df = pd.read_excel(raw_path, dtype={"InvoiceNo": str, "StockCode": str,
                                        "Description": str})
    raw_rows = len(df)
    raw_parquet = data_dir / "raw_online_retail.parquet"
    df.to_parquet(raw_parquet, compression="snappy", index=False)
    df = pd.read_parquet(raw_parquet)

    # ------------------------------------------------------------------
    # 2. Normalize and clean
    # ------------------------------------------------------------------
    print("[2/4] Cleaning")
    df = df.rename(columns={
        "InvoiceNo": "invoice_no", "StockCode": "stock_code",
        "Description": "description", "Quantity": "quantity",
        "InvoiceDate": "invoice_date", "UnitPrice": "unit_price",
        "CustomerID": "customer_id", "Country": "country"})

    df = df[df["invoice_no"].notna() & df["stock_code"].notna()
            & df["description"].notna()]
    df["invoice_no"] = df["invoice_no"].astype(str).str.strip()
    df = df[~df["invoice_no"].str.upper().str.startswith("C")]
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]

    df["description"] = (df["description"].astype(str).str.strip()
                         .str.replace(r"\s+", " ", regex=True))
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["basket_id"] = df["invoice_no"]
    df["revenue"] = df["quantity"] * df["unit_price"]

    # One product can carry slightly different descriptions across rows;
    # canonicalize to the most frequent description per stock_code so the
    # catalog, search, and recommendations all agree.
    canon = (df.groupby(["stock_code", "description"]).size()
               .reset_index(name="n")
               .sort_values("n", ascending=False)
               .drop_duplicates("stock_code")[["stock_code", "description"]])
    df = df.drop(columns=["description"]).merge(canon, on="stock_code", how="left")

    cleaned_rows = len(df)
    df.to_parquet(data_dir / "cleaned_order_lines.parquet",
                  compression="snappy", index=False)

    # ------------------------------------------------------------------
    # 3. Product lookup with popularity + synthetic demo fields
    # ------------------------------------------------------------------
    print("[3/4] Building product lookup")
    lookup = (df.groupby(["stock_code", "description"])
                .agg(total_units=("quantity", "sum"),
                     total_revenue=("revenue", "sum"),
                     order_count=("basket_id", "nunique"),
                     median_price=("unit_price", "median"))
                .reset_index())

    lookup["product_category"] = lookup["description"].map(assign_category)
    lookup["product_popularity_rank"] = (lookup["order_count"]
                                         .rank(ascending=False, method="first")
                                         .astype(int))

    # Price buckets from catalog-wide terciles of median price (demo field).
    q1, q2 = lookup["median_price"].quantile([1 / 3, 2 / 3])
    lookup["synthetic_price_bucket"] = lookup["median_price"].map(
        lambda p: "Budget" if p <= q1 else ("Mid" if p <= q2 else "Premium"))

    lookup["synthetic_margin_rate"] = lookup["product_category"].map(CATEGORY_MARGIN_RATE)

    # ~92% in stock, ~97% eligible — deterministic per SKU (stable hash).
    lookup["synthetic_in_stock"] = lookup["stock_code"].map(
        lambda s: stable_bucket(s + "|stock", 100) < 92)
    lookup["synthetic_eligibility_flag"] = lookup["stock_code"].map(
        lambda s: stable_bucket(s + "|elig", 100) < 97)

    lookup = lookup.drop(columns=["median_price"])
    lookup.to_parquet(art_dir / "product_lookup.parquet",
                      compression="snappy", index=False)

    # ------------------------------------------------------------------
    # 4. Summary log
    # ------------------------------------------------------------------
    print("[4/4] Done. Summary:")
    print(f"      Raw rows:      {raw_rows:,}")
    print(f"      Cleaned rows:  {cleaned_rows:,}")
    print(f"      Baskets:       {df['basket_id'].nunique():,}")
    print(f"      Products:      {lookup['stock_code'].nunique():,}")
    print(f"      Date range:    {df['invoice_date'].min().date()} -> {df['invoice_date'].max().date()}")
    print(f"      In stock:      {lookup['synthetic_in_stock'].mean():.0%} (synthetic)")
    print(f"      Eligible:      {lookup['synthetic_eligibility_flag'].mean():.0%} (synthetic)")
    print(f"      Outputs:       {data_dir} | {art_dir}")


if __name__ == "__main__":
    main()
