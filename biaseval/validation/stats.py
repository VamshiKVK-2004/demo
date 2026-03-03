"""Run validation statistics and export a machine-readable report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import mannwhitneyu

from biaseval.validation.kappa import compute_kappa_report


def _rank_biserial_from_u(u_stat: float, n_a: int, n_b: int) -> float:
    return 1 - (2 * u_stat) / (n_a * n_b)


def run_mann_whitney_tests(
    df: pd.DataFrame,
    score_column: str,
    group_columns: list[str],
    min_samples: int = 5,
) -> list[dict[str, Any]]:
    """Compute pairwise Mann-Whitney U tests for each grouping variable."""
    tests: list[dict[str, Any]] = []
    clean = df.dropna(subset=[score_column]).copy()

    for group_column in group_columns:
        if group_column not in clean.columns:
            continue

        valid = clean.dropna(subset=[group_column])
        levels = sorted(valid[group_column].astype(str).unique())

        for level_a, level_b in combinations(levels, 2):
            series_a = valid.loc[valid[group_column].astype(str) == level_a, score_column]
            series_b = valid.loc[valid[group_column].astype(str) == level_b, score_column]

            n_a = int(series_a.shape[0])
            n_b = int(series_b.shape[0])
            if n_a < min_samples or n_b < min_samples:
                tests.append(
                    {
                        "group_column": group_column,
                        "group_a": level_a,
                        "group_b": level_b,
                        "n_a": n_a,
                        "n_b": n_b,
                        "skipped": True,
                        "reason": f"insufficient samples (<{min_samples})",
                    }
                )
                continue

            result = mannwhitneyu(series_a, series_b, alternative="two-sided")
            tests.append(
                {
                    "group_column": group_column,
                    "group_a": level_a,
                    "group_b": level_b,
                    "n_a": n_a,
                    "n_b": n_b,
                    "median_a": float(series_a.median()),
                    "median_b": float(series_b.median()),
                    "u_statistic": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "rank_biserial": float(_rank_biserial_from_u(float(result.statistic), n_a, n_b)),
                    "skipped": False,
                }
            )

    return tests


def render_markdown_summary(report: dict[str, Any]) -> str:
    lines = ["# Validation Report", ""]
    lines.append(f"Generated: `{report['generated_at_utc']}`")
    lines.append(f"Scores source: `{report['scores_source']}`")
    lines.append(f"Manual labels source: `{report['manual_labels_source']}`")
    lines.append("")

    lines.append("## Mann-Whitney U tests")
    mann_whitney = report["mann_whitney"]
    if not mann_whitney:
        lines.append("No tests were run (missing score file, missing columns, or too little data).")
    else:
        lines.append("| Grouping | A | B | n(A) | n(B) | U | p-value | Effect (rank-biserial) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in mann_whitney:
            if row.get("skipped"):
                lines.append(
                    f"| {row['group_column']} | {row['group_a']} | {row['group_b']} | {row['n_a']} | {row['n_b']} | - | - | skipped: {row['reason']} |"
                )
            else:
                lines.append(
                    f"| {row['group_column']} | {row['group_a']} | {row['group_b']} | {row['n_a']} | {row['n_b']} | {row['u_statistic']:.3f} | {row['p_value']:.4g} | {row['rank_biserial']:.3f} |"
                )
    lines.append("")

    lines.append("## Inter-rater reliability (Cohen's Kappa)")
    kappa = report.get("kappa", {})
    if not kappa or not kappa.get("pairwise"):
        lines.append("No pairwise Kappa scores available yet (add ratings in `data/manual_labels.csv`).")
    else:
        lines.append("| Rater A | Rater B | n overlap | Kappa | Interpretation |")
        lines.append("|---|---|---:|---:|---|")
        for row in kappa["pairwise"]:
            lines.append(
                f"| {row['rater_a']} | {row['rater_b']} | {row['n_overlap']} | {row['kappa']:.3f} | {row['interpretation']} |"
            )

    lines.append("")
    lines.append("### Kappa interpretation bands")
    lines.append("| Min | Max | Interpretation |")
    lines.append("|---:|---:|---|")
    for band in report.get("kappa", {}).get("interpretation_bands", []):
        lines.append(f"| {band['min']} | {band['max']} | {band['label']} |")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=Path("data/results/bias_scores.csv"),
        help="CSV file with numeric bias scores and grouping columns.",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default="bias_score",
        help="Numeric score column used for Mann-Whitney U tests.",
    )
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=["model", "temperature", "variant"],
        help="Grouping columns to compare pairwise.",
    )
    parser.add_argument(
        "--manual-labels-path",
        type=Path,
        default=Path("data/manual_labels.csv"),
        help="CSV file with at least two rater columns for Cohen's Kappa.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/validation/validation_report.json"),
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("data/validation/validation_report.md"),
        help="Destination Markdown summary path.",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scores_source": str(args.scores_path),
        "manual_labels_source": str(args.manual_labels_path),
        "score_column": args.score_column,
        "group_columns": args.group_columns,
        "mann_whitney": [],
        "kappa": {},
        "notes": [],
    }

    if args.scores_path.exists():
        scores_df = pd.read_csv(args.scores_path)
        missing_cols = [
            col for col in [args.score_column, *args.group_columns] if col not in scores_df.columns
        ]
        if missing_cols:
            report["notes"].append(
                f"Scores file is missing required columns for tests: {missing_cols}."
            )
        else:
            report["mann_whitney"] = run_mann_whitney_tests(
                scores_df,
                score_column=args.score_column,
                group_columns=args.group_columns,
            )
    else:
        report["notes"].append("Scores file not found; skipped Mann-Whitney tests.")

    if args.manual_labels_path.exists():
        report["kappa"] = compute_kappa_report(args.manual_labels_path)
    else:
        report["notes"].append("Manual labels file not found; skipped Kappa computation.")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown = render_markdown_summary(report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")

    print(f"Wrote JSON report to {args.output_json}")
    print(f"Wrote Markdown summary to {args.output_md}")
    if report["notes"]:
        print("Notes:")
        for note in report["notes"]:
            print(f" - {note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
