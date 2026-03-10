"""Validation package."""

import json
from datetime import datetime, timezone
from pathlib import Path


def run() -> None:
    """Run validation stage and persist JSON/Markdown reports."""
    from biaseval.validation.kappa import compute_kappa_report
    from biaseval.validation.stats import render_markdown_summary, run_mann_whitney_tests
    import pandas as pd

    scores_path = Path("artifacts/metrics_bias_response.parquet")
    manual_labels_path = Path("data/manual_labels.csv")
    output_json = Path("data/validation/validation_report.json")
    output_md = Path("data/validation/validation_report.md")
    kappa_json = Path("data/validation/kappa_report.json")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scores_source": str(scores_path),
        "manual_labels_source": str(manual_labels_path),
        "score_column": "bias_score",
        "group_columns": ["model", "temperature", "variant"],
        "mann_whitney": [],
        "kappa": {},
        "notes": [],
    }

    if scores_path.exists():
        scores_df = pd.read_parquet(scores_path)
        required = ["bias_score", "model", "temperature", "variant"]
        missing = [col for col in required if col not in scores_df.columns]
        if missing:
            report["notes"].append(f"Scores file is missing required columns for tests: {missing}.")
        else:
            report["mann_whitney"] = run_mann_whitney_tests(
                scores_df,
                score_column="bias_score",
                group_columns=["model", "temperature", "variant"],
            )
    else:
        report["notes"].append("Scores file not found; skipped Mann-Whitney tests.")

    if manual_labels_path.exists():
        report["kappa"] = compute_kappa_report(manual_labels_path)
    else:
        report["notes"].append("Manual labels file not found; skipped Kappa computation.")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown_summary(report), encoding="utf-8")

    if report.get("kappa"):
        kappa_json.write_text(json.dumps(report["kappa"], indent=2), encoding="utf-8")

    print(f"[biaseval] wrote validation JSON to {output_json}")
    print(f"[biaseval] wrote validation Markdown to {output_md}")
    print(f"[biaseval] wrote kappa JSON to {kappa_json}")
