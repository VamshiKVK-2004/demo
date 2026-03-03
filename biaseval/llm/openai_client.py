"""OpenAI provider client with a unified generate interface."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


class OpenAIClient:
    """Minimal OpenAI chat completions wrapper."""

    def __init__(self, api_key: str | None = None, timeout_s: int = 60) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
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
                "error": "missing OPENAI_API_KEY",
                "raw": None,
            }

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed

        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
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
            text = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
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
