"""OpenAI provider for GPT-5.2 with automatic prompt caching."""

from __future__ import annotations

import json
import time

import openai

from . import LLMResponse

# Pricing per 1M tokens
OPENAI_PRICING = {
    "gpt-5.2": {
        "input": 2.50,
        "output": 10.0,
        "cached_input": 0.625,  # 75% discount
    },
}


class OpenAIProvider:
    def __init__(self, model_id: str = "gpt-5.2"):
        self.model_id = model_id
        self.client = openai.OpenAI()
        self._pricing = OPENAI_PRICING.get(model_id, OPENAI_PRICING["gpt-5.2"])

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
    ) -> LLMResponse:
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_id,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        latency_ms = (time.time() - start) * 1000

        raw_text = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        cached_tokens = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = getattr(cached_tokens, "cached_tokens", 0) if cached_tokens else 0

        non_cached_input = input_tokens - cached_tokens
        cost = (
            (non_cached_input / 1_000_000) * self._pricing["input"]
            + (cached_tokens / 1_000_000) * self._pricing["cached_input"]
            + (output_tokens / 1_000_000) * self._pricing["output"]
        )

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
    try:
        data = json.loads(raw_text)
        action_val = data["action"]
        reasoning = data.get("reasoning", "")
        scratchpad = data.get("scratchpad", "")

        # String actions: "trade", "accept", "reject"
        _TRADE_ACTIONS = {"trade": -1, "accept": -2, "reject": -3}
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

        action = int(action_val)
        return action, reasoning, scratchpad, None
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    for token in raw_text.split():
        try:
            return int(token), "", "", None
        except ValueError:
            continue

    return -1, raw_text, "", None
