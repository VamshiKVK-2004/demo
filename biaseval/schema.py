"""Schema definitions and validation for preprocessing artifacts."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

RAW_RESPONSE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "provider",
    "model",
    "temperature",
    "prompt_id",
    "variant",
    "response_text",
    "timestamp",
    "latency_ms",
    "error",
)

PROCESSED_RESPONSE_COLUMNS: tuple[str, ...] = (
    *RAW_RESPONSE_COLUMNS,
    "normalized_text",
    "sentences",
    "lemmas",
    "content_lemmas",
    "entities",
)


def _assert_columns(df: pd.DataFrame, required: Iterable[str], schema_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{schema_name} schema missing required columns: {missing}")


def validate_raw_response_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize raw response schema from the LLM stage."""
    _assert_columns(df, RAW_RESPONSE_COLUMNS, "raw response")
    return df.loc[:, RAW_RESPONSE_COLUMNS].copy()


def validate_processed_response_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize processed response schema for downstream stages."""
    _assert_columns(df, PROCESSED_RESPONSE_COLUMNS, "processed response")
    return df.loc[:, PROCESSED_RESPONSE_COLUMNS].copy()
