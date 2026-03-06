"""Hugging Face provider client with a unified generate interface."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class HuggingFaceClient:
    """Minimal Hugging Face Inference API wrapper for text generation models."""

    def __init__(self, api_key: str | None = None, timeout_s: int = 60) -> None:
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        self.timeout_s = timeout_s

    def generate(self, prompt: str, model: str, temperature: float, seed: int | None = None) -> dict[str, Any]:
        """Generate text for a prompt.

        Returns a standard response schema:
        {
            "response_text": str,
            "latency_ms": int,
            "error": str | None,
            "raw": dict | None,
        }
        """
        start = time.perf_counter()
        if not self.api_key:
            return {
                "response_text": "",
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "error": "missing HUGGINGFACE_API_KEY/HF_TOKEN",
                "raw": None,
            }

        endpoint = f"https://api-inference.huggingface.co/models/{urllib.parse.quote(model)}"
        payload: dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "return_full_text": False,
            },
        }
        if seed is not None:
            payload["parameters"]["seed"] = seed

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))

            text = ""
            if isinstance(raw, list) and raw:
                candidate = raw[0] or {}
                text = candidate.get("generated_text", "")
            elif isinstance(raw, dict):
                text = raw.get("generated_text", "")

            return {
                "response_text": text,
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "error": None,
                "raw": raw,
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return {
                "response_text": "",
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "error": f"http_{exc.code}: {body}",
                "raw": None,
            }
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            return {
                "response_text": "",
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "error": str(exc),
                "raw": None,
            }
