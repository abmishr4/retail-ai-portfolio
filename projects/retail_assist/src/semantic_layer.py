"""
Retail Assist — semantic_layer.py

Loads the governed configuration (configs/metrics.yml) and exposes it as the
single gatekeeper for what the app may compute. Every metric or dimension
used anywhere downstream (router, SQL templates, LLM layer) must pass
through validate_request(). Anything outside this layer is unsupported by
design — that is the core governance idea this app demonstrates.
"""

from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent      # projects/retail_assist
CONFIG_PATH = BASE_DIR / "configs" / "metrics.yml"


class SemanticLayer:
    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.metrics: dict = cfg["metrics"]
        self.dimensions: dict = cfg["dimensions"]
        self.unsupported_domains: list = cfg["unsupported_domains"]
        self.data_info: dict = cfg["data"]

    # ------------------------------------------------------------------
    # Allowlists
    # ------------------------------------------------------------------
    def allowed_metrics(self) -> list:
        return list(self.metrics.keys())

    def allowed_dimensions(self) -> list:
        return list(self.dimensions.keys())

    # ------------------------------------------------------------------
    # Lookups (raise KeyError on anything outside the governed config —
    # failing loudly is intentional)
    # ------------------------------------------------------------------
    def metric_expression(self, metric: str) -> str:
        return self.metrics[metric]["expression"]

    def metric_format(self, metric: str) -> str:
        return self.metrics[metric].get("format", "number")

    def metric_description(self, metric: str) -> str:
        return self.metrics[metric]["description"]

    def dimension_column(self, dimension: str) -> str:
        return self.dimensions[dimension]["column"]

    # ------------------------------------------------------------------
    # Validation — the gate everything must pass
    # ------------------------------------------------------------------
    def validate_request(self, metrics: list, dimensions: list) -> tuple[bool, str]:
        """Return (ok, reason). ok=False means the request is unsupported."""
        for m in metrics:
            if m not in self.metrics:
                return False, f"'{m}' is not a supported metric. Supported: {', '.join(self.allowed_metrics())}."
        for d in dimensions:
            if d not in self.dimensions:
                return False, f"'{d}' is not a supported dimension. Supported: {', '.join(self.allowed_dimensions())}."
        return True, ""

    # ------------------------------------------------------------------
    # For UI display (Semantic Layer tab) and the refusal message
    # ------------------------------------------------------------------
    def definitions_table(self) -> list:
        rows = []
        for name, m in self.metrics.items():
            rows.append({
                "name": name,
                "type": "metric",
                "expression": m["expression"],
                "aggregation_rule": m["aggregation_rule"],
                "description": m["description"],
            })
        for name, d in self.dimensions.items():
            rows.append({
                "name": name,
                "type": "dimension",
                "expression": d["column"],
                "aggregation_rule": "group_by",
                "description": d["description"],
            })
        return rows

    def refusal_message(self) -> str:
        return (
            "I cannot answer this from the governed demo data because this "
            "question is outside the supported metric and dimension set. "
            f"Supported metrics: {', '.join(self.allowed_metrics())}. "
            f"Supported dimensions: {', '.join(self.allowed_dimensions())}."
        )
