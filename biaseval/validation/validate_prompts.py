"""Validate prompt assets for completeness and category balance."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_VARIANTS = {"neutral", "biased", "counterfactual"}
REQUIRED_COLUMNS = {
    "prompt_id",
    "base_prompt_id",
    "theme",
    "variant",
    "target_group",
    "counterfactual_group",
    "prompt_text",
    "notes",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON prompt file must contain a top-level array of records.")
        return data
    raise ValueError(f"Unsupported prompt file format: {path.suffix}")


def validate_required_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["Prompt file is empty."]
    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        return [f"Missing required columns: {sorted(missing)}"]
    return []


def validate_unique_prompt_ids(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    counts = Counter(row["prompt_id"] for row in rows)
    duplicates = sorted(prompt_id for prompt_id, count in counts.items() if count > 1)
    if duplicates:
        errors.append(f"Duplicate prompt_id values found: {duplicates[:10]}")
    return errors


def validate_variant_triplets(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    variants_by_base: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        variants_by_base[row["base_prompt_id"]].add(row["variant"])

    for base_prompt_id, variants in sorted(variants_by_base.items()):
        if variants != REQUIRED_VARIANTS:
            errors.append(
                f"Base scenario {base_prompt_id} has variants {sorted(variants)}; "
                f"expected {sorted(REQUIRED_VARIANTS)}"
            )
    return errors


def validate_balanced_coverage(rows: list[dict[str, str]], tolerance: int) -> list[str]:
    errors: list[str] = []
    neutral_rows = [row for row in rows if row["variant"] == "neutral"]
    if not neutral_rows:
        return ["No neutral prompts found; cannot evaluate base-category balance."]

    by_theme = Counter(row["theme"] for row in neutral_rows)
    if max(by_theme.values()) - min(by_theme.values()) > tolerance:
        errors.append(f"Theme imbalance detected for neutral prompts: {dict(by_theme)}")

    groups_within_theme: dict[str, Counter[str]] = defaultdict(Counter)
    for row in neutral_rows:
        groups_within_theme[row["theme"]][row["target_group"]] += 1

    for theme, counts in sorted(groups_within_theme.items()):
        if counts and max(counts.values()) - min(counts.values()) > tolerance:
            errors.append(
                f"Target-group imbalance in theme '{theme}': {dict(counts)}"
            )

    return errors


def validate_rows(rows: list[dict[str, str]], tolerance: int) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_required_columns(rows))
    if errors:
        return errors
    errors.extend(validate_unique_prompt_ids(rows))
    errors.extend(validate_variant_triplets(rows))
    errors.extend(validate_balanced_coverage(rows, tolerance=tolerance))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("data/prompts/base_prompts.csv"),
        help="Path to prompts file (CSV or JSON).",
    )
    parser.add_argument(
        "--balance-tolerance",
        type=int,
        default=0,
        help="Maximum allowed spread between smallest and largest category counts.",
    )
    args = parser.parse_args()

    rows = load_rows(args.path)
    errors = validate_rows(rows, tolerance=args.balance_tolerance)
    if errors:
        print("Prompt validation failed:")
        for error in errors:
            print(f" - {error}")
        return 1

    print(
        f"Validation passed for {args.path} with {len(rows)} prompt variants "
        f"({len(rows) // len(REQUIRED_VARIANTS)} base scenarios)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
