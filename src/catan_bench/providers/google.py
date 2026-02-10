"""Google provider for Gemini models."""

from __future__ import annotations

import json
import logging
import time

from google import genai
from google.genai import types

from . import LLMResponse

logger = logging.getLogger(__name__)

# Pricing per 1M tokens
GOOGLE_PRICING = {
    "gemini-3-pro-preview": {
        "input": 1.25,
        "output": 5.0,
    },
    "gemini-3-flash-preview": {
        "input": 0.50,
        "output": 3.0,
    },
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.0,
    },
    "gemini-2.5-flash": {
        "input": 0.30,
        "output": 2.50,
    },
}

# Minimum thinking budget for Gemini 2.5+ thinking models.
# Added on top of max_output_tokens so visible response keeps full budget.
_THINKING_BUDGET = 128

# Disable safety filters — this is a board game, not harmful content
_SAFETY_OFF = [
    types.SafetySetting(category=c, threshold="OFF")
    for c in [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
]


class GoogleProvider:
    def __init__(self, model_id: str = "gemini-3-pro-preview"):
        self.model_id = model_id
        self._pricing = GOOGLE_PRICING.get(model_id, GOOGLE_PRICING["gemini-3-pro-preview"])
        # Client reads GEMINI_API_KEY or GOOGLE_API_KEY from env
        self._client = genai.Client()

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
    ) -> LLMResponse:
        # Thinking models (2.5/3 series) burn thinking tokens from
        # max_output_tokens. Cap thinking at 128 (minimum for 2.5-pro)
        # and add that on top so the visible response keeps the full budget.
        start = time.time()
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens + _THINKING_BUDGET,
                thinking_config=types.ThinkingConfig(thinking_budget=_THINKING_BUDGET),
                safety_settings=_SAFETY_OFF,
            ),
        )
        latency_ms = (time.time() - start) * 1000

        raw_text = response.text or ""

        # Extract token usage
        usage = response.usage_metadata
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0

        non_cached_input = input_tokens - cached_tokens
        cost = (
            (non_cached_input / 1_000_000) * self._pricing["input"]
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
    text = raw_text.strip()

    # Extract from code block if present
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:  # odd-indexed parts are inside code blocks
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

    logger.warning(f"Failed to parse response: {raw_text[:300]}")
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
