"""
Retail Assist — data_prep.py

One-time offline preparation. Converts the raw UCI Online Retail Excel file
into small, governed parquet artifacts that the Streamlit app reads at runtime.
The deployed app NEVER touches the raw Excel.

Run (from repo root):
    python projects/retail_assist/src/data_prep.py
Or point at any file/folder explicitly:
    python data_prep.py --raw "/path/to/Online Retail.xlsx" --out "/path/to/processed"

Synthetic enrichment disclosure:
    product_category, margin_rate, gross_margin, channel, promo_flag,
    promo_depth, customer_segment, and fulfillment_type are SYNTHETIC
    demo fields. They are deterministic (stable-hash / rule based, no
    randomness) so results are fully reproducible. They do not represent
    any real retailer's business logic.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default repo paths (overridable via CLI)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # projects/retail_assist
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_OUT_DIR = BASE_DIR / "data" / "processed"

RANDOM_SEED = 42  # kept for reproducibility of any future sampled logic

# ---------------------------------------------------------------------------
# Synthetic enrichment rules (deterministic, clearly demo-only)
# ---------------------------------------------------------------------------
# Keyword -> category map applied to product description (first match wins).
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

# Synthetic margin rate by category (demo values, not real economics).
CATEGORY_MARGIN_RATE = {
    "Christmas & Seasonal": 0.52,
    "Candles & Fragrance":  0.55,
    "Kitchen & Dining":     0.48,
    "Bags & Accessories":   0.50,
    "Stationery & Crafts":  0.58,
    "Toys & Games":         0.45,
    "Garden & Outdoor":     0.42,
    "Home Decor":           0.50,
    DEFAULT_CATEGORY:       0.46,
}

CHANNELS = ["Web", "Mobile App", "Marketplace"]              # invoice-level
FULFILLMENT_TYPES = ["Ship to Home", "Store Pickup", "Locker"]  # invoice-level
PROMO_DEPTHS = [0.10, 0.15, 0.20, 0.25]                      # when promo_flag = 1


def stable_bucket(key: str, buckets: int) -> int:
    """Deterministic 0..buckets-1 assignment from a string key.

    Uses md5 (stable across runs/machines) rather than Python's salted hash()
    or RNG, so every rerun of this script yields identical synthetic fields.
    """
    digest = hashlib.md5(str(key).encode("utf-8")).hexdigest()
    return int(digest, 16) % buckets


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
            f"No .xlsx found in {DEFAULT_RAW_DIR}. "
            "Download the UCI Online Retail dataset and place the Excel file there, "
            "or pass --raw /path/to/file.xlsx"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail Assist data prep")
    parser.add_argument("--raw", default=None, help="Path to the raw UCI Excel file")
    parser.add_argument("--out", default=None, help="Output folder for parquet artifacts")
    args = parser.parse_args()

    raw_path = find_raw_file(args.raw)
    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_SEED)

    # -----------------------------------------------------------------
    # 1. Load raw Excel and immediately persist as parquet
    # -----------------------------------------------------------------
    print(f"[1/6] Loading raw Excel: {raw_path}")
    df = pd.read_excel(raw_path, dtype={"InvoiceNo": str, "StockCode": str, "Description": str})
    raw_rows = len(df)

    raw_parquet = out_dir / "raw_online_retail.parquet"
    df.to_parquet(raw_parquet, compression="snappy", index=False)
    print(f"      Raw rows: {raw_rows:,}  ->  {raw_parquet.name}")

    # All subsequent processing reads from parquet, not Excel.
    df = pd.read_parquet(raw_parquet)

    # -----------------------------------------------------------------
    # 2. Normalize column names
    # -----------------------------------------------------------------
    print("[2/6] Normalizing columns and cleaning rows")
    df = df.rename(columns={
        "InvoiceNo": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "UnitPrice": "unit_price",
        "CustomerID": "customer_id",
        "Country": "country",
    })

    # -----------------------------------------------------------------
    # 3. Remove invalid rows
    # -----------------------------------------------------------------
    df = df[df["invoice_no"].notna()]
    df = df[df["stock_code"].notna()]
    df = df[df["description"].notna()]
    df["invoice_no"] = df["invoice_no"].astype(str).str.strip()
    df = df[~df["invoice_no"].str.upper().str.startswith("C")]   # cancellations
    df = df[df["quantity"] > 0]
    df = df[df["unit_price"] > 0]

    df["description"] = df["description"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    # customer_id: keep valid invoice rows even when null; label "Guest"
    # so order-level metrics (revenue, orders, AOV, ...) stay complete.
    df["customer_id"] = df["customer_id"].apply(
        lambda x: f"{int(x)}" if pd.notna(x) else "Guest"
    )

    # -----------------------------------------------------------------
    # 4. Derived + synthetic enrichment fields (all deterministic)
    # -----------------------------------------------------------------
    print("[3/6] Building revenue and deterministic synthetic enrichments")
    df["revenue"] = df["quantity"] * df["unit_price"]

    # Product-level: category (keyword map) and margin (by category)
    df["product_category"] = df["description"].map(assign_category)
    df["margin_rate"] = df["product_category"].map(CATEGORY_MARGIN_RATE)
    df["gross_margin"] = df["revenue"] * df["margin_rate"]

    # Invoice-level synthetic attributes (an order has ONE channel /
    # fulfillment / promo state), assigned by stable hash of invoice_no.
    invoices = pd.DataFrame({"invoice_no": df["invoice_no"].unique()})
    invoices["channel"] = invoices["invoice_no"].map(
        lambda i: CHANNELS[stable_bucket(i + "|ch", len(CHANNELS))]
    )
    invoices["fulfillment_type"] = invoices["invoice_no"].map(
        lambda i: FULFILLMENT_TYPES[stable_bucket(i + "|ff", len(FULFILLMENT_TYPES))]
    )
    # ~30% of invoices flagged as promo, deterministic per invoice.
    invoices["promo_flag"] = invoices["invoice_no"].map(
        lambda i: 1 if stable_bucket(i + "|pr", 100) < 30 else 0
    )
    invoices["promo_depth"] = invoices.apply(
        lambda r: PROMO_DEPTHS[stable_bucket(r["invoice_no"] + "|pd", len(PROMO_DEPTHS))]
        if r["promo_flag"] == 1 else 0.0,
        axis=1,
    )
    df = df.merge(invoices, on="invoice_no", how="left")

    # -----------------------------------------------------------------
    # 5. Customer segments (RFM-style, identified customers only)
    # -----------------------------------------------------------------
    print("[4/6] Assigning RFM-style customer segments")
    anchor_date = df["invoice_date"].max() + pd.Timedelta(days=1)
    known = df[df["customer_id"] != "Guest"]
    rfm = known.groupby("customer_id").agg(
        recency_days=("invoice_date", lambda s: (anchor_date - s.max()).days),
        frequency=("invoice_no", "nunique"),
        monetary=("revenue", "sum"),
    ).reset_index()

    freq_hi = rfm["frequency"].quantile(0.75)
    mon_hi = rfm["monetary"].quantile(0.75)
    rec_recent = rfm["recency_days"].quantile(0.50)

    def segment(row):
        recent = row["recency_days"] <= rec_recent
        if recent and (row["frequency"] >= freq_hi or row["monetary"] >= mon_hi):
            return "High Value"
        if recent:
            return "Loyal"
        if row["frequency"] >= freq_hi or row["monetary"] >= mon_hi:
            return "At Risk"      # was valuable, hasn't purchased recently
        return "Lapsed"

    rfm["customer_segment"] = rfm.apply(segment, axis=1)
    seg_map = dict(zip(rfm["customer_id"], rfm["customer_segment"]))
    df["customer_segment"] = df["customer_id"].map(seg_map).fillna("Guest")

    cleaned_rows = len(df)

    # -----------------------------------------------------------------
    # 6. Write governed artifacts
    # -----------------------------------------------------------------
    print("[5/6] Writing parquet artifacts")
    df["month"] = df["invoice_date"].dt.to_period("M").astype(str)

    fact_cols = [
        "invoice_no", "stock_code", "description", "quantity", "invoice_date",
        "month", "unit_price", "revenue", "customer_id", "country",
        "product_category", "margin_rate", "gross_margin",
        "channel", "promo_flag", "promo_depth", "customer_segment", "fulfillment_type",
    ]
    fact_order_lines = df[fact_cols].copy()
    fact_order_lines.to_parquet(out_dir / "fact_order_lines.parquet",
                                compression="snappy", index=False)

    # Daily rollup at INVOICE-SAFE grain only (all dims are invoice-level),
    # so COUNT(DISTINCT invoice_no) remains valid and AOV stays correct.
    # Category/product/segment questions query fact_order_lines directly.
    df["order_date"] = df["invoice_date"].dt.date
    fact_daily_metrics = (
        df.groupby(["order_date", "country", "channel", "promo_flag", "fulfillment_type"])
          .agg(revenue=("revenue", "sum"),
               gross_margin=("gross_margin", "sum"),
               units=("quantity", "sum"),
               orders=("invoice_no", "nunique"))
          .reset_index()
    )
    fact_daily_metrics.to_parquet(out_dir / "fact_daily_metrics.parquet",
                                  compression="snappy", index=False)

    dim_product = (
        df.groupby(["stock_code", "description", "product_category", "margin_rate"])
          .agg(total_revenue=("revenue", "sum"),
               total_units=("quantity", "sum"),
               n_orders=("invoice_no", "nunique"))
          .reset_index()
    )
    dim_product.to_parquet(out_dir / "dim_product.parquet",
                           compression="snappy", index=False)

    dim_customer = rfm[["customer_id", "recency_days", "frequency",
                        "monetary", "customer_segment"]].copy()
    dim_customer.to_parquet(out_dir / "dim_customer.parquet",
                            compression="snappy", index=False)

    cal = pd.DataFrame({"date": pd.date_range(df["invoice_date"].min().normalize(),
                                              df["invoice_date"].max().normalize(),
                                              freq="D")})
    cal["month"] = cal["date"].dt.to_period("M").astype(str)
    cal["year"] = cal["date"].dt.year
    cal["week"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["day_name"] = cal["date"].dt.day_name()
    cal.to_parquet(out_dir / "dim_calendar.parquet",
                   compression="snappy", index=False)

    # -----------------------------------------------------------------
    # Summary log
    # -----------------------------------------------------------------
    print("[6/6] Done. Summary:")
    print(f"      Raw rows:        {raw_rows:,}")
    print(f"      Cleaned rows:    {cleaned_rows:,}")
    print(f"      Products:        {dim_product['stock_code'].nunique():,}")
    print(f"      Invoices:        {fact_order_lines['invoice_no'].nunique():,}")
    print(f"      Customers:       {dim_customer['customer_id'].nunique():,} identified (+ Guest)")
    print(f"      Date range:      {df['invoice_date'].min().date()} -> {df['invoice_date'].max().date()}")
    print(f"      Output folder:   {out_dir}")
    for p in sorted(out_dir.glob("*.parquet")):
        print(f"        {p.name:35s} {p.stat().st_size/1024/1024:6.2f} MB")


if __name__ == "__main__":
    main()
