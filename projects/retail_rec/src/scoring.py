"""
Retail Rec — scoring.py

Turns raw co-purchase statistics into a single base score, with all weights
loaded from configs/scoring_config.yml (never hardcoded in Python).

Normalization design (documented because it IS the judgment call):
  * lift is heavy-tailed — a pair seen in one basket can show lift in the
    hundreds. Min-max on raw lift would let those outliers flatten every
    real signal to ~0. So lift and co_purchase_count are log-scaled
    (log1p) BEFORE min-max normalization.
  * confidence is already bounded [0,1]; plain min-max.
  * All three normalized components land in [0,1], so the YAML weights are
    directly interpretable as relative importance.

base_score = w_lift * normalized_lift
           + w_confidence * normalized_confidence
           + w_count * normalized_log_co_purchase_count
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent          # projects/retail_rec
DEFAULT_CONFIG = BASE_DIR / "configs" / "scoring_config.yml"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def score_candidates(cand: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add normalized components and base_score. Returns a copy."""
    w = cfg["retrieval_weights"]
    out = cand.copy()

    out["normalized_lift"] = _minmax(np.log1p(out["lift"]))
    out["normalized_confidence"] = _minmax(out["confidence"])
    out["normalized_log_co_purchase_count"] = _minmax(
        np.log1p(out["co_purchase_count"]))

    out["base_score"] = (
        w["lift"] * out["normalized_lift"]
        + w["confidence"] * out["normalized_confidence"]
        + w["co_purchase_count"] * out["normalized_log_co_purchase_count"]
    ).round(4)
    return out
