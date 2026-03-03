"""LLM inference package."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from biaseval.llm.gemini_client import GeminiClient
from biaseval.llm.openai_client import OpenAIClient
from biaseval.schema import validate_raw_response_schema

TEMPERATURES = [0.0, 0.3, 0.7]
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.5
MIN_INTERVAL_BY_PROVIDER_S = {
    "openai": 0.75,
    "gemini": 1.0,
}


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_experiments(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config.get("experiments", [])


def _persist_results(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = validate_raw_response_schema(pd.DataFrame(rows))
    parquet_path = output_dir / "raw_responses.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        jsonl_path = output_dir / "raw_responses.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        return jsonl_path


def run() -> None:
    """Execute prompts against configured providers and persist raw outputs."""
    prompts = _load_prompts(Path("data/prompts/base_prompts.json"))
    experiments = _load_experiments(Path("config/experiments.yaml"))

    clients = {
        "openai": OpenAIClient(),
        "gemini": GeminiClient(),
    }
    last_call_at: dict[str, float] = {provider: 0.0 for provider in clients}

    run_id = uuid.uuid4().hex
    rows: list[dict[str, Any]] = []

    for experiment in experiments:
        provider = experiment.get("provider")
        model = experiment.get("model", "")
        if provider not in clients:
            print(f"[biaseval] skipping unknown provider: {provider}")
            continue

        client = clients[provider]
        min_interval_s = MIN_INTERVAL_BY_PROVIDER_S.get(provider, 0.5)

        for prompt in prompts:
            prompt_text = prompt.get("prompt_text", "")
            for temperature in TEMPERATURES:
                now = time.perf_counter()
                elapsed = now - last_call_at[provider]
                if elapsed < min_interval_s:
                    time.sleep(min_interval_s - elapsed)

                result = None
                for attempt in range(MAX_RETRIES):
                    result = client.generate(
                        prompt=prompt_text,
                        model=model,
                        temperature=temperature,
                        seed=42,
                    )
                    if not result.get("error"):
                        break
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(BACKOFF_BASE_S**attempt)

                last_call_at[provider] = time.perf_counter()
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                rows.append(
                    {
                        "run_id": run_id,
                        "provider": provider,
                        "model": model,
                        "temperature": temperature,
                        "prompt_id": prompt.get("prompt_id"),
                        "variant": prompt.get("variant"),
                        "response_text": (result or {}).get("response_text", ""),
                        "timestamp": timestamp,
                        "latency_ms": (result or {}).get("latency_ms", 0),
                        "error": (result or {}).get("error"),
                    }
                )

    artifact = _persist_results(rows, Path("artifacts"))
    print(f"[biaseval] wrote {len(rows)} raw responses to {artifact}")
