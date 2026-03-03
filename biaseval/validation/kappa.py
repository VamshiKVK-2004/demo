"""Compute Cohen's Kappa agreement for manual annotation labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

INTERPRETATION_BANDS = [
    (-1.0, 0.0, "poor"),
    (0.0, 0.20, "slight"),
    (0.21, 0.40, "fair"),
    (0.41, 0.60, "moderate"),
    (0.61, 0.80, "substantial"),
    (0.81, 1.00, "almost perfect"),
]


def interpret_kappa(value: float) -> str:
    """Map a kappa score to Landis-Koch interpretation bands."""
    for low, high, label in INTERPRETATION_BANDS:
        if low <= value <= high:
            return label
    if value < -1:
        return "below theoretical range"
    return "above theoretical range"


def cohens_kappa(labels_a: pd.Series, labels_b: pd.Series) -> float:
    """Compute Cohen's Kappa for two categorical label vectors."""
    paired = pd.DataFrame({"a": labels_a, "b": labels_b}).dropna()
    if paired.empty:
        raise ValueError("No overlapping annotations after dropping missing labels.")

    confusion = pd.crosstab(paired["a"], paired["b"])
    n_obs = confusion.to_numpy().sum()
    observed_agreement = confusion.to_numpy().trace() / n_obs

    row_marginals = confusion.sum(axis=1) / n_obs
    col_marginals = confusion.sum(axis=0) / n_obs
    expected_agreement = float((row_marginals.to_numpy()[:, None] * col_marginals.to_numpy()[None, :]).sum())

    if expected_agreement == 1:
        return 1.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def compute_pairwise_kappas(df: pd.DataFrame, rater_columns: list[str]) -> list[dict[str, Any]]:
    """Compute kappa for all rater pairs with non-empty overlap."""
    results: list[dict[str, Any]] = []
    for i, rater_a in enumerate(rater_columns):
        for rater_b in rater_columns[i + 1 :]:
            paired = df[[rater_a, rater_b]].dropna()
            if paired.empty:
                continue
            score = cohens_kappa(paired[rater_a], paired[rater_b])
            results.append(
                {
                    "rater_a": rater_a,
                    "rater_b": rater_b,
                    "n_overlap": int(len(paired)),
                    "kappa": float(score),
                    "interpretation": interpret_kappa(float(score)),
                }
            )
    return results


def compute_kappa_report(path: Path) -> dict[str, Any]:
    """Load manual labels and compute pairwise Cohen's Kappa metrics."""
    df = pd.read_csv(path)
    rater_columns = [column for column in df.columns if column.lower().startswith("rater_")]
    if len(rater_columns) < 2:
        raise ValueError("At least two rater columns are required (e.g., rater_1, rater_2).")

    pairwise = compute_pairwise_kappas(df, rater_columns)
    return {
        "source": str(path),
        "rater_columns": rater_columns,
        "pairwise": pairwise,
        "interpretation_bands": [
            {"min": low, "max": high, "label": label}
            for low, high, label in INTERPRETATION_BANDS
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("data/manual_labels.csv"),
        help="Path to manual annotation CSV with at least two rater columns.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to export a JSON report.",
    )
    args = parser.parse_args()

    report = compute_kappa_report(args.path)
    print(json.dumps(report, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
