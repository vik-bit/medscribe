"""LLM backends. Free options only: a local Ollama server, or Groq's free tier.

All backends return an :class:`LLMResult` carrying exact token counts reported
by the server, so cost comparisons between pipelines use measured tokens,
not estimates.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class LLM(Protocol):
    name: str

    def complete(
        self, system: str, user: str, *, schema: dict[str, Any] | None, timeout_s: float
    ) -> LLMResult: ...


class OllamaLLM:
    """Ollama ``/api/chat`` with structured output (``format`` = JSON schema)."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str | None = None,
        num_ctx: int = 8192,
        max_output_tokens: int = 1536,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        # num_predict caps runaway generations (a 3B model can loop inside a JSON
        # list); a truncated reply fails validation and is retried like any other.
        self.options = {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "seed": seed,
            "num_predict": max_output_tokens,
        }
        self.name = f"ollama:{model}"

    def complete(self, system, user, *, schema=None, timeout_s=300.0) -> LLMResult:
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": self.options,
        }
        if schema is not None:
            body["format"] = schema
        t0 = time.perf_counter()
        r = httpx.post(f"{self.host}/api/chat", json=body, timeout=timeout_s)
        r.raise_for_status()
        d = r.json()
        return LLMResult(
            text=d["message"]["content"],
            prompt_tokens=int(d.get("prompt_eval_count", 0)),
            completion_tokens=int(d.get("eval_count", 0)),
            latency_s=time.perf_counter() - t0,
        )


class GroqLLM:
    """Groq OpenAI-compatible endpoint (free tier). Needs ``GROQ_API_KEY``."""

    def __init__(self, model: str = "llama-3.1-8b-instant", temperature: float = 0.0) -> None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self._headers = {"Authorization": f"Bearer {key}"}
        self.model = model
        self.temperature = temperature
        self.name = f"groq:{model}"

    def complete(self, system, user, *, schema=None, timeout_s=120.0) -> LLMResult:
        if schema is not None:
            user = f"{user}\n\nReply with JSON only, matching this schema:\n{json.dumps(schema)}"
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if schema is not None:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        r = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=body,
            headers=self._headers,
            timeout=timeout_s,
        )
        r.raise_for_status()
        d = r.json()
        usage = d.get("usage", {})
        return LLMResult(
            text=d["choices"][0]["message"]["content"],
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=time.perf_counter() - t0,
        )


class ScriptedLLM:
    """Deterministic test double: ``responder(system, user) -> str``."""

    def __init__(self, responder: Callable[[str, str], str], name: str = "scripted") -> None:
        self.responder = responder
        self.name = name
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, *, schema=None, timeout_s=30.0) -> LLMResult:
        self.calls.append((system, user))
        text = self.responder(system, user)
        return LLMResult(text, len(user) // 4, len(text) // 4, 0.0)


class FaultInjectingLLM:
    """Wraps a backend and injects failures at a fixed, seeded rate.

    Fault kinds: ``timeout`` (raises :class:`TimeoutError`), ``malformed``
    (returns truncated JSON), ``empty`` (returns ``""``). Used to test that the
    orchestrator degrades gracefully instead of crashing.
    """

    KINDS = ("timeout", "malformed", "empty")

    def __init__(self, inner: LLM, rate: float, seed: int = 0, kinds=KINDS) -> None:
        self.inner = inner
        self.rate = rate
        self.kinds = tuple(kinds)
        self.rng = random.Random(seed)
        self.name = f"faulty({rate}):{inner.name}"
        self.injected: list[str] = []

    def complete(self, system, user, *, schema=None, timeout_s=300.0) -> LLMResult:
        if self.rng.random() < self.rate:
            kind = self.rng.choice(self.kinds)
            self.injected.append(kind)
            if kind == "timeout":
                raise TimeoutError("injected timeout")
            res = self.inner.complete(system, user, schema=schema, timeout_s=timeout_s)
            res.text = "" if kind == "empty" else res.text[: max(1, len(res.text) // 2)]
            return res
        return self.inner.complete(system, user, schema=schema, timeout_s=timeout_s)


def make_llm(spec: str) -> LLM:
    """``ollama:qwen2.5:3b`` or ``groq:llama-3.1-8b-instant``."""
    kind, _, model = spec.partition(":")
    if kind == "ollama":
        return OllamaLLM(model or "qwen2.5:3b")
    if kind == "groq":
        return GroqLLM(model or "llama-3.1-8b-instant")
    raise ValueError(f"unknown LLM spec {spec!r}")
