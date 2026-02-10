"""LLM provider interfaces and cost tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class LLMResponse:
    """Parsed response from an LLM provider."""
    action_index: int
    reasoning: str
    raw_text: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    latency_ms: float
    cost_usd: float
    scratchpad: str = ""
    trade_proposal: dict | None = None  # populated when action is "trade"


@dataclass
class CostTracker:
    """Accumulates cost and token usage across calls."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0
    total_calls: int = 0
    total_latency_ms: float = 0.0

    def record(self, response: LLMResponse) -> None:
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cached_tokens += response.cached_tokens
        self.total_cost_usd += response.cost_usd
        self.total_calls += 1
        self.total_latency_ms += response.latency_ms

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.total_calls, 1)

    @property
    def cache_hit_rate(self) -> float:
        total = self.total_input_tokens + self.total_cached_tokens
        if total == 0:
            return 0.0
        return self.total_cached_tokens / total


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    model_id: str

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
    ) -> LLMResponse:
        """Send a prompt and return the parsed response."""
        ...

