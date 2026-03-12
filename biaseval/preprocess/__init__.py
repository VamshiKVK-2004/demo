"""Text preprocessing stage for deterministic response normalization."""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import pandas as pd
import spacy

from biaseval.schema import (
    PROCESSED_RESPONSE_COLUMNS,
    validate_processed_response_schema,
    validate_raw_response_schema,
)

RAW_ARTIFACT_PATH = Path("artifacts/raw_responses.parquet")
PROCESSED_ARTIFACT_PATH = Path("artifacts/processed_responses.parquet")


def _parse_flag(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.lower().split())


def _build_nlp(extract_entities: bool):
    disable_components = ["parser"]
    if not extract_entities:
        disable_components.append("ner")

    nlp = spacy.load("en_core_web_sm", disable=disable_components)
    if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def _process_row(row: pd.Series, nlp, extract_entities: bool) -> dict[str, object]:
    response_text = str(row["response_text"])
    normalized_text = _normalize(response_text)
    doc = nlp(normalized_text)

    lemmas = [token.lemma_.lower() for token in doc if not token.is_space]
    content_lemmas = [
        token.lemma_.lower()
        for token in doc
        if not token.is_space and not token.is_stop and not token.is_punct
    ]

    entities: list[dict[str, str]] = []
    if extract_entities and "ner" in nlp.pipe_names:
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    return {
        **row.to_dict(),
        "normalized_text": normalized_text,
        "sentences": [sent.text.strip() for sent in doc.sents if sent.text.strip()],
        "lemmas": lemmas,
        "content_lemmas": content_lemmas,
        "entities": entities,
    }


def _summarize_invalid_rows(raw_df: pd.DataFrame) -> str:
    error_mask = raw_df["error"].fillna("").astype(str).str.strip().ne("")
    empty_response_mask = raw_df["response_text"].fillna("").astype(str).str.strip().eq("")
    both_mask = error_mask & empty_response_mask

    summary_lines = [
        f"total rows={len(raw_df)}",
        f"rows with provider errors={int(error_mask.sum())}",
        f"rows with empty responses={int(empty_response_mask.sum())}",
        f"rows with both issues={int(both_mask.sum())}",
    ]

    provider_breakdown = (
        raw_df.loc[error_mask]
        .groupby("provider", dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    if not provider_breakdown.empty:
        formatted = ", ".join(
            f"{provider}: {count}" for provider, count in provider_breakdown.items()
        )
        summary_lines.append(f"error rows by provider={formatted}")

    top_errors = (
        raw_df.loc[error_mask, "error"]
        .astype(str)
        .str.strip()
        .value_counts()
        .head(3)
    )
    if not top_errors.empty:
        formatted = "; ".join(f"{message} ({count})" for message, count in top_errors.items())
        summary_lines.append(f"top provider error messages={formatted}")

    return " | ".join(summary_lines)


def run() -> None:
    """Load raw responses, preprocess deterministically, and persist processed artifact."""
    if not RAW_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Raw artifact not found at {RAW_ARTIFACT_PATH}. Run the llm stage first."
        )

    extract_entities = _parse_flag(os.getenv("BIASEVAL_EXTRACT_ENTITIES"), default=False)

    raw_df = pd.read_parquet(RAW_ARTIFACT_PATH)
    raw_df = validate_raw_response_schema(raw_df)
    diagnostics = _summarize_invalid_rows(raw_df)

    valid_mask = (
        raw_df["error"].fillna("").astype(str).str.strip().eq("")
        & raw_df["response_text"].fillna("").astype(str).str.strip().ne("")
    )
    dropped = int((~valid_mask).sum())
    if dropped:
        print(
            f"[biaseval] dropping {dropped} invalid raw rows "
            "(missing/empty responses or provider errors) before preprocessing"
        )
    raw_df = raw_df.loc[valid_mask].copy()
    if raw_df.empty:
        print(f"[biaseval] preprocess diagnostics: {diagnostics}")
        raise ValueError(
            "No valid raw responses available after filtering errors/empty outputs. "
            "Check provider credentials and collect-stage logs. "
            f"Diagnostics: {diagnostics}."
        )

    nlp = _build_nlp(extract_entities=extract_entities)
    processed_rows = [
        _process_row(row=row, nlp=nlp, extract_entities=extract_entities)
        for _, row in raw_df.iterrows()
    ]

    processed_df = pd.DataFrame(processed_rows)
    processed_df = validate_processed_response_schema(processed_df)

    PROCESSED_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(PROCESSED_ARTIFACT_PATH, index=False)

    columns = ", ".join(PROCESSED_RESPONSE_COLUMNS)
    print(
        f"[biaseval] wrote {len(processed_df)} processed responses to {PROCESSED_ARTIFACT_PATH} "
        f"(entities={'on' if extract_entities else 'off'}; columns={columns})"
    )
