"""
Retail Assist — sql_templates.py

Static, pre-written, parameterized SQL templates executed with DuckDB
against the governed parquet artifacts.

Governance rules enforced here:
  * SQL is NEVER generated dynamically or by an LLM. Every query is built
    from pre-written skeletons below.
  * Metric expressions come only from the semantic layer (metrics.yml).
  * Dimension names are validated against the semantic layer allowlist
    before they touch a query. User text never enters SQL.
  * AOV and margin_rate are computed as weighted ratios at query time —
    never averaged across groups.

Every template returns a plan dict:
  template_name, sql, metrics_used, dimensions_used, time_window, filters
so the UI can show the full query plan for transparency.
"""

from pathlib import Path

import duckdb

from semantic_layer import SemanticLayer

BASE_DIR = Path(__file__).resolve().parent.parent      # projects/retail_assist
DATA_DIR = BASE_DIR / "data" / "processed"
FACT_PATH = DATA_DIR / "fact_order_lines.parquet"

# Fixed dataset window (UCI Online Retail). "last month" means the last
# COMPLETE month in the data — Nov 2011, since December 2011 is partial.
LAST_COMPLETE_MONTH = "2011-11"
PRIOR_MONTH = "2011-10"

TIME_WINDOWS = {
    "all_time": ("All available data (Dec 2010 – Dec 9, 2011)", ""),
    "last_month": (f"Last complete month in data ({LAST_COMPLETE_MONTH})",
                   f"WHERE month = '{LAST_COMPLETE_MONTH}'"),
    "mom": (f"Month-over-month ({PRIOR_MONTH} vs {LAST_COMPLETE_MONTH})",
            f"WHERE month IN ('{PRIOR_MONTH}', '{LAST_COMPLETE_MONTH}')"),
}


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        f"CREATE OR REPLACE VIEW orders AS SELECT * FROM read_parquet('{FACT_PATH.as_posix()}')"
    )
    return con


def _select_clause(sem: SemanticLayer, metrics: list) -> str:
    parts = [f"{sem.metric_expression(m)} AS {m}" for m in metrics]
    return ",\n       ".join(parts)


def build_plan(
    sem: SemanticLayer,
    template_name: str,
    metrics: list,
    dimensions: list,
    time_window: str = "all_time",
    top_n: int = 10,
    order_by: str | None = None,
) -> dict:
    """Build a validated query plan from one of the static skeletons.

    Raises ValueError if anything falls outside the governed config —
    the router converts that into a safe refusal.
    """
    ok, reason = sem.validate_request(metrics, dimensions)
    if not ok:
        raise ValueError(reason)
    if time_window not in TIME_WINDOWS:
        raise ValueError(f"Unsupported time window '{time_window}'.")
    if order_by is not None and order_by not in metrics:
        raise ValueError("order_by must be one of the selected metrics.")

    window_label, where = TIME_WINDOWS[time_window]
    select_metrics = _select_clause(sem, metrics)

    if template_name == "overall_summary":
        # Single-row totals, no grouping.
        sql = f"SELECT {select_metrics}\nFROM orders\n{where}".strip()

    elif template_name == "metric_by_dimension":
        # The workhorse: metric(s) grouped by ONE governed dimension.
        dim_col = sem.dimension_column(dimensions[0])
        order_metric = order_by or metrics[0]
        sql = (
            f"SELECT {dim_col} AS {dimensions[0]},\n"
            f"       {select_metrics}\n"
            f"FROM orders\n{where}\n"
            f"GROUP BY {dim_col}\n"
            f"ORDER BY {order_metric} DESC"
        ).strip()

    elif template_name == "trend_by_month":
        # Time series, always month-ordered ascending.
        sql = (
            f"SELECT month,\n"
            f"       {select_metrics}\n"
            f"FROM orders\n{where}\n"
            f"GROUP BY month\n"
            f"ORDER BY month ASC"
        ).strip()

    elif template_name == "top_n_products":
        order_metric = order_by or metrics[0]
        sql = (
            f"SELECT description AS product,\n"
            f"       product_category,\n"
            f"       {select_metrics}\n"
            f"FROM orders\n{where}\n"
            f"GROUP BY description, product_category\n"
            f"ORDER BY {order_metric} DESC\n"
            f"LIMIT {int(top_n)}"
        ).strip()

    else:
        raise ValueError(f"Unknown SQL template '{template_name}'.")

    return {
        "template_name": template_name,
        "sql": sql,
        "metrics_used": metrics,
        "dimensions_used": dimensions,
        "time_window": window_label,
        "filters": where.replace("WHERE ", "") if where else "none",
        "top_n": top_n if template_name == "top_n_products" else None,
    }


def run_plan(plan: dict):
    """Execute a plan built by build_plan(). Returns a pandas DataFrame."""
    con = _connect()
    try:
        return con.execute(plan["sql"]).df()
    finally:
        con.close()
