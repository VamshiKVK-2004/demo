"""LLM inference package."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from biaseval.llm.gemini_client import GeminiClient
from biaseval.llm.huggingface_client import HuggingFaceClient
from biaseval.llm.openai_client import OpenAIClient
from biaseval.schema import RAW_RESPONSE_COLUMNS, validate_raw_response_schema

TEMPERATURES = [0.0, 0.3, 0.7]
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.5
DEFAULT_MIN_INTERVAL_BY_PROVIDER_S = {
    "openai": 0.5,
    "gemini": 0.25,
    "huggingface": 0.5,
}


def _provider_has_credentials(provider: str) -> bool:
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if provider == "gemini":
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if provider == "huggingface":
        return bool(os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN"))
    return False


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_experiments(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config.get("experiments", [])


def _float_env(name: str) -> float | None:
    value = os.getenv(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        print(f"[biaseval] ignoring invalid {name}={value!r}; expected float seconds")
        return None


def _min_interval_seconds(provider: str, experiment: dict[str, Any]) -> float:
    if "min_interval_s" in experiment:
        try:
            return max(0.0, float(experiment["min_interval_s"]))
        except (TypeError, ValueError):
            print(
                f"[biaseval] ignoring invalid min_interval_s={experiment['min_interval_s']!r}"
                f" for provider {provider}"
            )

    provider_env = _float_env(f"BIASEVAL_MIN_INTERVAL_{provider.upper()}_S")
    if provider_env is not None:
        return max(0.0, provider_env)

    global_env = _float_env("BIASEVAL_MIN_INTERVAL_S")
    if global_env is not None:
        return max(0.0, global_env)

    return DEFAULT_MIN_INTERVAL_BY_PROVIDER_S.get(provider, 0.25)


def _max_prompts_limit() -> int | None:
    value = os.getenv("BIASEVAL_MAX_PROMPTS")
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        print(f"[biaseval] ignoring invalid BIASEVAL_MAX_PROMPTS={value!r}; expected int")
        return None
    return max(1, parsed)


def _persist_results(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        df = validate_raw_response_schema(pd.DataFrame(rows))
    else:
        df = pd.DataFrame(columns=RAW_RESPONSE_COLUMNS)
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
    prompt_limit = _max_prompts_limit()
    if prompt_limit is not None and prompt_limit < len(prompts):
        prompts = prompts[:prompt_limit]
        print(f"[biaseval] limiting collect stage to first {prompt_limit} prompts (BIASEVAL_MAX_PROMPTS)")

    experiments = _load_experiments(Path("config/experiments.yaml"))

    clients = {
        "openai": OpenAIClient(),
        "gemini": GeminiClient(),
        "huggingface": HuggingFaceClient(),
    }
    last_call_at: dict[str, float] = {provider: 0.0 for provider in clients}

    run_id = uuid.uuid4().hex
    rows: list[dict[str, Any]] = []

    total_requests = len(prompts) * len(TEMPERATURES) * len(experiments)
    completed_requests = 0

    for experiment in experiments:
        provider = experiment.get("provider")
        model = experiment.get("model", "")
        if provider not in clients:
            print(f"[biaseval] skipping unknown provider: {provider}")
            continue
        if not _provider_has_credentials(provider):
            print(f"[biaseval] skipping provider {provider}: missing credentials")
            continue

        client = clients[provider]
        min_interval_s = _min_interval_seconds(provider, experiment)
        print(
            f"[biaseval] collect provider={provider} model={model} prompts={len(prompts)} "
            f"temps={TEMPERATURES} min_interval_s={min_interval_s}"
        )

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
                    error = (result or {}).get("error") or ""
                    if not error:
                        break
                    if error.startswith("missing "):
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
                completed_requests += 1
                if completed_requests % 25 == 0:
                    print(f"[biaseval] collect progress: {completed_requests}/{total_requests} requests")

    artifact = _persist_results(rows, Path("artifacts"))
    print(f"[biaseval] wrote {len(rows)} raw responses to {artifact}")
