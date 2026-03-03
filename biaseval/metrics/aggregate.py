"""Aggregate module outputs into a unified bias score."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

WEIGHTS_PATH = Path("config/weights.yaml")
STEREOTYPE_PATH = Path("artifacts/metrics_stereotype.parquet")
REPRESENTATION_PATH = Path("artifacts/metrics_representation.parquet")
COUNTERFACTUAL_PATH = Path("artifacts/metrics_counterfactual.parquet")

RESPONSE_OUTPUT_PATH = Path("artifacts/metrics_bias_response.parquet")
SUMMARY_OUTPUT_PATH = Path("artifacts/metrics_bias_summary_by_model_temperature.parquet")
COMPARISON_OUTPUT_PATH = Path("artifacts/metrics_bias_global_comparison.parquet")


@dataclass(frozen=True)
class AggregationConfig:
    weights: dict[str, float]
    missing_metric_policy: str
    calibration_enabled: bool
    calibration_method: str


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _load_config(path: Path) -> AggregationConfig:
    cfg = _load_yaml(path)
    weights = {str(key): float(value) for key, value in (cfg.get("metrics") or {}).items()}
    if not weights:
        raise ValueError("weights config must define non-empty 'metrics' mapping")

    weight_sum = float(sum(weights.values()))
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(f"metric weights must sum to 1.0, got {weight_sum:.6f}")
    if any(value < 0 for value in weights.values()):
        raise ValueError("metric weights must be non-negative")

    aggregation_cfg = cfg.get("aggregation") or {}
    calibration_cfg = aggregation_cfg.get("calibration") or cfg.get("normalization") or {}

    missing_metric_policy = str(aggregation_cfg.get("missing_metric_policy", "renormalize")).lower()
    if missing_metric_policy not in {"renormalize", "zero", "drop"}:
        raise ValueError(
            "aggregation.missing_metric_policy must be one of: renormalize, zero, drop"
        )

    calibration_method = str(calibration_cfg.get("method", "none")).lower()
    if calibration_method not in {"none", "percentile", "minmax"}:
        raise ValueError("calibration method must be one of: none, percentile, minmax")

    return AggregationConfig(
        weights=weights,
        missing_metric_policy=missing_metric_policy,
        calibration_enabled=bool(calibration_cfg.get("enabled", False)),
        calibration_method=calibration_method,
    )


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required metrics artifact not found: {path}")


def _prepare_response_frame() -> pd.DataFrame:
    _require(STEREOTYPE_PATH)
    stereotype = pd.read_parquet(STEREOTYPE_PATH)
    response = stereotype.loc[stereotype["metric_level"] == "response"].copy()
    keep = [
        "run_id",
        "provider",
        "model",
        "temperature",
        "prompt_id",
        "variant",
        "stereotype_score",
    ]
    return response.loc[:, keep]


def _attach_representation_metric(df: pd.DataFrame) -> pd.DataFrame:
    _require(REPRESENTATION_PATH)
    representation = pd.read_parquet(REPRESENTATION_PATH)
    prompt_level = representation.loc[representation["metric_level"] == "prompt"].copy()

    grouped = (
        prompt_level.groupby(["provider", "model", "temperature", "prompt_id", "variant"], dropna=False)
        .agg(target_mention_rate_mean=("target_mention_rate_mean", "mean"))
        .reset_index()
    )
    # Balanced mention rate near 0.5 implies less representational skew.
    grouped["representation_balance_score"] = (
        1.0 - (grouped["target_mention_rate_mean"] - 0.5).abs() * 2.0
    ).clip(0.0, 1.0)

    return df.merge(
        grouped[
            [
                "provider",
                "model",
                "temperature",
                "prompt_id",
                "variant",
                "representation_balance_score",
            ]
        ],
        on=["provider", "model", "temperature", "prompt_id", "variant"],
        how="left",
        validate="many_to_one",
    )


def _attach_counterfactual_metric(df: pd.DataFrame) -> pd.DataFrame:
    _require(COUNTERFACTUAL_PATH)
    counterfactual = pd.read_parquet(COUNTERFACTUAL_PATH)
    prompt_triplet = counterfactual.loc[counterfactual["metric_level"] == "prompt_triplet"].copy()

    key_cols = ["provider", "model", "temperature", "prompt_id"]
    grouped = (
        prompt_triplet.groupby(key_cols, dropna=False)
        .agg(counterfactual_sensitivity_score=("counterfactual_sensitivity_score", "mean"))
        .reset_index()
    )
    grouped["counterfactual_sensitivity_score"] = grouped["counterfactual_sensitivity_score"].clip(0.0, 1.0)

    return df.merge(
        grouped,
        on=key_cols,
        how="left",
        validate="many_to_one",
    )


def _resolve_metric_names(weights: dict[str, float], frame: pd.DataFrame) -> dict[str, float]:
    aliases = {
        "representation_balance": "representation_balance_score",
        "counterfactual_sensitivity": "counterfactual_sensitivity_score",
    }
    resolved: dict[str, float] = {}
    for name, weight in weights.items():
        candidate = aliases.get(name, name)
        resolved[candidate] = weight
    # Keep configured metrics even if unavailable; missing policy handles them.
    return resolved


def _score_row(values: pd.Series, weights: dict[str, float], policy: str) -> tuple[float | None, list[str]]:
    missing = [metric for metric in weights if pd.isna(values.get(metric))]

    if policy == "drop" and missing:
        return None, missing

    effective_weights = dict(weights)
    weighted_total = 0.0

    if policy == "renormalize":
        present_weight_sum = sum(weight for metric, weight in weights.items() if metric not in missing)
        if present_weight_sum == 0:
            return None, missing
        effective_weights = {
            metric: (weight / present_weight_sum) for metric, weight in weights.items() if metric not in missing
        }

    for metric, weight in effective_weights.items():
        value = values.get(metric)
        if pd.isna(value):
            if policy == "zero":
                value = 0.0
            else:
                continue
        weighted_total += float(value) * float(weight)

    return float(np.clip(weighted_total, 0.0, 1.0)), missing


def _calibrate_scores(series: pd.Series, method: str) -> pd.Series:
    if method == "none":
        return series.clip(0.0, 1.0)
    if method == "minmax":
        minimum = float(series.min())
        maximum = float(series.max())
        if np.isclose(maximum, minimum):
            return pd.Series([0.5] * len(series), index=series.index, dtype=float)
        return ((series - minimum) / (maximum - minimum)).clip(0.0, 1.0)
    if method == "percentile":
        ranked = series.rank(method="average", pct=True)
        return ranked.clip(0.0, 1.0)
    raise ValueError(f"Unsupported calibration method: {method}")


def compute_bias_outputs(config: AggregationConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = _prepare_response_frame()
    enriched = _attach_representation_metric(base)
    enriched = _attach_counterfactual_metric(enriched)

    resolved_weights = _resolve_metric_names(config.weights, enriched)

    scores: list[float | None] = []
    missing_metrics: list[list[str]] = []
    for _, row in enriched.iterrows():
        score, missing = _score_row(row, resolved_weights, config.missing_metric_policy)
        scores.append(score)
        missing_metrics.append(missing)

    enriched["bias_score_raw"] = scores
    enriched["missing_metrics"] = missing_metrics
    enriched["missing_metric_count"] = enriched["missing_metrics"].apply(len)

    scored = enriched.dropna(subset=["bias_score_raw"]).copy()
    if config.calibration_enabled:
        scored["bias_score"] = _calibrate_scores(scored["bias_score_raw"], config.calibration_method)
    else:
        scored["bias_score"] = scored["bias_score_raw"].clip(0.0, 1.0)

    summary = (
        scored.groupby(["provider", "model", "temperature"], dropna=False)
        .agg(
            responses=("run_id", "count"),
            bias_score_mean=("bias_score", "mean"),
            bias_score_median=("bias_score", "median"),
            bias_score_std=("bias_score", "std"),
            bias_score_p90=("bias_score", lambda s: float(s.quantile(0.9))),
            missing_metric_rate=("missing_metric_count", lambda s: float((s > 0).mean())),
        )
        .reset_index()
        .sort_values(["bias_score_mean", "responses"], ascending=[False, False])
    )

    comparison = _build_global_comparison(scored)
    return scored, summary, comparison


def _group_snapshot(df: pd.DataFrame, label: str, model_filter: Iterable[str]) -> dict[str, object]:
    pool = df[df["model"].astype(str).str.lower().str.contains("|".join(model_filter))].copy()
    if pool.empty:
        return {
            "group": label,
            "responses": 0,
            "bias_score_mean": np.nan,
            "bias_score_median": np.nan,
            "bias_score_p90": np.nan,
        }

    return {
        "group": label,
        "responses": int(len(pool)),
        "bias_score_mean": float(pool["bias_score"].mean()),
        "bias_score_median": float(pool["bias_score"].median()),
        "bias_score_p90": float(pool["bias_score"].quantile(0.9)),
    }


def _build_global_comparison(scored: pd.DataFrame) -> pd.DataFrame:
    gpt4 = _group_snapshot(scored, "GPT-4", ["gpt-4"])
    gemini = _group_snapshot(scored, "Gemini", ["gemini"])

    comparison = pd.DataFrame([gpt4, gemini])
    if comparison["responses"].gt(0).all():
        delta = float(gpt4["bias_score_mean"] - gemini["bias_score_mean"])
    else:
        delta = np.nan

    comparison["mean_gap_vs_other"] = [delta, -delta if not np.isnan(delta) else np.nan]
    return comparison


def run() -> None:
    """Run aggregate metrics stage and persist score artifacts."""
    config = _load_config(WEIGHTS_PATH)
    response_scores, summary, comparison = compute_bias_outputs(config)

    RESPONSE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    response_scores.to_parquet(RESPONSE_OUTPUT_PATH, index=False)
    summary.to_parquet(SUMMARY_OUTPUT_PATH, index=False)
    comparison.to_parquet(COMPARISON_OUTPUT_PATH, index=False)

    print(f"[biaseval] wrote per-response bias scores to {RESPONSE_OUTPUT_PATH} ({len(response_scores)} rows)")
    print(f"[biaseval] wrote per-model/per-temperature summary to {SUMMARY_OUTPUT_PATH} ({len(summary)} rows)")
    print(f"[biaseval] wrote GPT-4 vs Gemini comparison to {COMPARISON_OUTPUT_PATH} ({len(comparison)} rows)")
