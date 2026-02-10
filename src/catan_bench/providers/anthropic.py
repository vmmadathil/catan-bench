"""Anthropic provider for Claude Opus 4.6 and Sonnet 4.5 with prompt caching."""

from __future__ import annotations

import json
import time

import anthropic

from . import LLMResponse

# Pricing per 1M tokens (as of 2025)
ANTHROPIC_PRICING = {
    "claude-opus-4-6": {
        "input": 15.0,
        "output": 75.0,
        "cached_input": 1.5,  # 90% discount
    },
    "claude-sonnet-4-5-20250929": {
        "input": 3.0,
        "output": 15.0,
        "cached_input": 0.3,
    },
    "claude-haiku-4-5-20251001": {
        "input": 0.80,
        "output": 4.0,
        "cached_input": 0.08,
    },
}


class AnthropicProvider:
    def __init__(self, model_id: str = "claude-sonnet-4-5-20250929"):
        self.model_id = model_id
        self.client = anthropic.Anthropic()
        self._pricing = ANTHROPIC_PRICING.get(model_id, ANTHROPIC_PRICING["claude-sonnet-4-5-20250929"])

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
    ) -> LLMResponse:
        start = time.time()
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        latency_ms = (time.time() - start) * 1000

        raw_text = response.content[0].text
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

        # Non-cached input tokens = total input - cached read
        non_cached_input = input_tokens - cached_tokens
        cost = (
            (non_cached_input / 1_000_000) * self._pricing["input"]
            + (cached_tokens / 1_000_000) * self._pricing["cached_input"]
            + (output_tokens / 1_000_000) * self._pricing["output"]
        )

        # Parse action index from JSON response
        action_index, reasoning, scratchpad, trade_proposal = _parse_response(raw_text)

        return LLMResponse(
            action_index=action_index,
            reasoning=reasoning,
            raw_text=raw_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            scratchpad=scratchpad,
            trade_proposal=trade_proposal,
        )

def _parse_response(raw_text: str) -> tuple[int, str, str, dict | None]:
    """Parse action index, reasoning, scratchpad, and optional trade proposal."""
    text = raw_text.strip()

    # Extract from code block if present
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            result = _try_parse_json(part)
            if result is not None:
                return result

    # Try to find JSON by locating first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        result = _try_parse_json(text[start : end + 1])
        if result is not None:
            return result

    # Try the whole text as JSON
    result = _try_parse_json(text)
    if result is not None:
        return result

    # Fallback: look for bare integer
    for token in raw_text.split():
        try:
            return int(token), "", "", None
        except ValueError:
            continue

    # Last resort
    return -1, raw_text, "", None


# Sentinel action indices for trade responses
TRADE_PROPOSE = -1
TRADE_ACCEPT = -2
TRADE_REJECT = -3

_TRADE_ACTIONS = {"trade": TRADE_PROPOSE, "accept": TRADE_ACCEPT, "reject": TRADE_REJECT}


def _try_parse_json(text: str) -> tuple[int, str, str, dict | None] | None:
    """Try to parse JSON with an 'action' field. Returns None on failure."""
    try:
        data = json.loads(text)
        action_val = data["action"]
        reasoning = data.get("reasoning", "")
        scratchpad = data.get("scratchpad", "")

        # String actions: "trade", "accept", "reject"
        if isinstance(action_val, str) and action_val.lower() in _TRADE_ACTIONS:
            action_key = action_val.lower()
            sentinel = _TRADE_ACTIONS[action_key]
            trade_proposal = None
            if action_key == "trade":
                trade_proposal = {
                    "target": data.get("target", ""),
                    "offering": data.get("offering", {}),
                    "requesting": data.get("requesting", {}),
                }
            return sentinel, reasoning, scratchpad, trade_proposal

        # Numeric action (normal flow)
        action = int(action_val)
        return action, reasoning, scratchpad, None
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None
