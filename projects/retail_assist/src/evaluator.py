"""
Retail Assist — evaluator.py

Structured benchmark over the labeled eval set (eval/eval_questions.csv).
Mirrors the evaluation-first philosophy the app demonstrates: routing is
scored against expected labels, not eyeballed.

Scoring:
  routing accuracy        predicted intent == expected intent
  metric selection        every expected metric is present in the detected
                          set (extra sensible defaults are not penalized)
  dimension selection     detected breakdown dimensions == expected (set eq)
  time-window accuracy    exact match
  refusal accuracy        unsupported questions must be refused
  semantic correctness    ALL of the above per question
                          (supported: route+metrics+dims+window+rows>0;
                           unsupported: refused)

Run (from repo root):
    python projects/retail_assist/src/evaluator.py
Writes eval/eval_results.csv and prints a summary.
"""

from pathlib import Path

import pandas as pd

from semantic_layer import SemanticLayer
from query_router import route_question, plan_from_route
from sql_templates import run_plan

BASE_DIR = Path(__file__).resolve().parent.parent      # projects/retail_assist
EVAL_DIR = BASE_DIR / "eval"
QUESTIONS_PATH = EVAL_DIR / "eval_questions.csv"
RESULTS_PATH = EVAL_DIR / "eval_results.csv"


def _split(cell) -> set:
    if pd.isna(cell) or str(cell).strip() == "":
        return set()
    return {t.strip() for t in str(cell).split(";") if t.strip()}


def run_eval(questions_path: Path = QUESTIONS_PATH,
             results_path: Path = RESULTS_PATH) -> pd.DataFrame:
    sem = SemanticLayer()
    qs = pd.read_csv(questions_path)
    rows = []

    for _, q in qs.iterrows():
        route = route_question(sem, q["question"])
        supported_expected = str(q["should_answer"]).strip().lower() == "yes"

        intent_ok = route["intent"] == q["expected_intent"]

        if supported_expected:
            metrics_ok = _split(q["expected_metrics"]).issubset(set(route["metrics"]))
            dims_ok = set(route["dimensions"]) == _split(q["expected_dimensions"])
            window_ok = route["time_window"] == q["expected_time_window"]
            refused_ok = route["intent"] != "unsupported"

            execution_ok = False
            if refused_ok and intent_ok:
                plan = plan_from_route(sem, route)
                if plan is not None:
                    df = run_plan(plan)
                    execution_ok = df is not None and len(df) > 0
            semantic_ok = all([intent_ok, metrics_ok, dims_ok, window_ok, execution_ok])
        else:
            # For unsupported questions the ONLY correct behavior is refusal.
            metrics_ok = dims_ok = window_ok = True
            refused_ok = route["intent"] == "unsupported"
            execution_ok = refused_ok
            semantic_ok = refused_ok

        rows.append({
            "question": q["question"],
            "should_answer": q["should_answer"],
            "expected_intent": q["expected_intent"],
            "predicted_intent": route["intent"],
            "predicted_metrics": ";".join(route["metrics"]),
            "predicted_dimensions": ";".join(route["dimensions"]),
            "predicted_time_window": route["time_window"],
            "confidence": route["confidence"],
            "intent_ok": intent_ok,
            "metrics_ok": metrics_ok,
            "dimensions_ok": dims_ok,
            "time_window_ok": window_ok,
            "refusal_ok": refused_ok,
            "execution_ok": execution_ok,
            "semantic_ok": semantic_ok,
        })

    results = pd.DataFrame(rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)

    sup = results[results["should_answer"].str.lower() == "yes"]
    uns = results[results["should_answer"].str.lower() == "no"]

    summary = {
        "questions_total": len(results),
        "routing_accuracy": results["intent_ok"].mean(),
        "metric_selection_accuracy": sup["metrics_ok"].mean(),
        "dimension_selection_accuracy": sup["dimensions_ok"].mean(),
        "time_window_accuracy": sup["time_window_ok"].mean(),
        "unsupported_refusal_accuracy": uns["refusal_ok"].mean(),
        "overall_semantic_correctness": results["semantic_ok"].mean(),
    }

    print("Retail Assist — evaluation summary")
    print("-" * 44)
    for k, v in summary.items():
        val = f"{v:.1%}" if k != "questions_total" else f"{v}"
        print(f"{k:32s} {val}")
    fails = results[~results["semantic_ok"]]
    if len(fails):
        print("\nFailures:")
        for _, f in fails.iterrows():
            print(f"  [{f.expected_intent} -> {f.predicted_intent}] {f.question}")
    print(f"\nResults written to {results_path}")
    return results


if __name__ == "__main__":
    run_eval()
