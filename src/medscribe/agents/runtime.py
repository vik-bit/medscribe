"""Agent runtime: budgets, traces, timeouts and parse-retry for typed LLM calls."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from medscribe.agents.llm import LLM

M = TypeVar("M", bound=BaseModel)


class BudgetExceeded(RuntimeError):
    pass


class AgentFailed(RuntimeError):
    pass


@dataclass
class Budget:
    """Hard limits for one pipeline run. Checked before every LLM call."""

    max_tokens: int = 60_000
    max_steps: int = 12
    max_seconds: float = 1800.0
    tokens: int = 0
    steps: int = 0
    started: float = field(default_factory=time.perf_counter)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def check(self) -> None:
        if self.steps >= self.max_steps:
            raise BudgetExceeded(f"step budget {self.max_steps} exhausted")
        if self.tokens >= self.max_tokens:
            raise BudgetExceeded(f"token budget {self.max_tokens} exhausted")
        if self.elapsed() >= self.max_seconds:
            raise BudgetExceeded(f"time budget {self.max_seconds}s exhausted")

    def remaining_seconds(self) -> float:
        return max(1.0, self.max_seconds - self.elapsed())


@dataclass
class TraceEvent:
    agent: str
    event: str  # call | ok | parse_error | timeout | error | info
    t: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    detail: str = ""


@dataclass
class Trace:
    events: list[TraceEvent] = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)

    def add(self, agent: str, event: str, **kw: Any) -> None:
        self.events.append(TraceEvent(agent, event, round(time.perf_counter() - self.t0, 3), **kw))

    def per_agent(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for e in self.events:
            a = out.setdefault(
                e.agent,
                {
                    "calls": 0,
                    "failures": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_s": 0.0,
                },
            )
            if e.event in ("ok", "parse_error", "timeout", "error"):
                a["calls"] += 1
                a["prompt_tokens"] += e.prompt_tokens
                a["completion_tokens"] += e.completion_tokens
                a["latency_s"] = round(a["latency_s"] + e.latency_s, 3)
            if e.event in ("parse_error", "timeout", "error"):
                a["failures"] += 1
        return out

    def to_json(self) -> list[dict[str, Any]]:
        return [e.__dict__ for e in self.events]


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_model(text: str, model: type[M]) -> M:
    """Parse ``text`` (JSON, possibly wrapped in a code fence) into ``model``."""
    cleaned = _FENCE.sub("", text.strip())
    if not cleaned.startswith("{"):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return model.model_validate(json.loads(cleaned))


@dataclass
class Context:
    llm: LLM
    budget: Budget
    trace: Trace
    call_timeout_s: float = 300.0
    parse_retries: int = 1


def call_typed(ctx: Context, agent: str, system: str, user: str, model: type[M]) -> M:
    """One typed LLM call with budget checks, timeout, tracing and parse-retry.

    Each attempt (including retries) counts as a step and its tokens are
    charged to the budget, so retries are never free.
    """
    schema = model.model_json_schema()
    last_err = ""
    for attempt in range(ctx.parse_retries + 1):
        ctx.budget.check()
        ctx.budget.steps += 1
        ctx.trace.add(agent, "call", detail=f"attempt {attempt + 1}")
        timeout = min(ctx.call_timeout_s, ctx.budget.remaining_seconds())
        try:
            res = ctx.llm.complete(system, user, schema=schema, timeout_s=timeout)
        except TimeoutError as e:
            ctx.trace.add(agent, "timeout", detail=str(e)[:200])
            last_err = f"timeout: {e}"
            continue
        except Exception as e:  # noqa: BLE001 - network/HTTP errors (httpx.TimeoutException etc.)
            kind = "timeout" if "timeout" in type(e).__name__.lower() else "error"
            ctx.trace.add(agent, kind, detail=f"{type(e).__name__}: {str(e)[:200]}")
            last_err = f"{kind}: {e}"
            continue
        ctx.budget.tokens += res.prompt_tokens + res.completion_tokens
        usage = dict(
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            latency_s=round(res.latency_s, 3),
        )
        try:
            out = parse_model(res.text, model)
        except (ValueError, ValidationError) as e:
            ctx.trace.add(agent, "parse_error", detail=str(e)[:200], **usage)
            last_err = f"parse: {str(e)[:200]}"
            user = (
                f"{user}\n\nYour previous reply was not valid JSON for the schema "
                f"({str(e)[:160]}). Reply again with valid JSON only."
            )
            continue
        ctx.trace.add(agent, "ok", **usage)
        return out
    raise AgentFailed(f"{agent}: {last_err}")
