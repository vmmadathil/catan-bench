# Catan Bench Backlog

## TODO
- [ ] Fix trade proposal/response parsing — still getting empty string errors from `complete_trade()`. Likely same thinking token issue (needs `thinking_config` in trade calls) or models returning non-JSON preamble. Check if `_extract_json` handles all cases.
- [ ] Visual board input mode (render board as image, pass to multimodal models)
- [ ] Add OpenAI/GPT models to benchmark config (note: GPT-4o has 200 RPD — too low for a 24-game run)
- [ ] Trace analysis script for safety/alignment review (see design below)
- [ ] Suppress `google_genai` AFC log noise
- [ ] Ablation: run with scratchpad/events toggled off to measure impact on ELO

## Design: Scratchpad for Multi-Turn Planning

**Problem:** Each LLM call is stateless — no memory of past turns, no ability to plan across turns.

**Solution:** A per-player scratchpad string that persists across turns within a game. The model reads its previous notes and writes updated notes each turn.

### Data flow
1. `LLMPlayer` holds `self._scratchpad: str = ""` (reset between games)
2. Scratchpad is appended to the **user prompt** (not system prompt — it changes every turn so can't be cached)
3. Response format expands to: `{"action": <index>, "reasoning": "...", "scratchpad": "..."}`
4. After parsing, `LLMPlayer` stores the new scratchpad for next turn
5. Cap scratchpad at ~500 tokens (~2000 chars) to control cost — if model exceeds, truncate to last N lines

### Files to modify
- **`llm_player.py`**: Add `_scratchpad` field, pass to prompt builder, extract from response, reset between games
- **`state_serializer.py`**: `build_user_prompt()` takes optional `scratchpad` param, appends section before the action list
- **`providers/anthropic.py` + `google.py`**: `_parse_response()` extracts `scratchpad` field from JSON, returns it alongside action/reasoning
- **`providers/__init__.py`**: Add `scratchpad` field to `LLMResponse`

### Prompt addition (in user prompt)
```
YOUR NOTES (from previous turns):
<scratchpad contents or "No notes yet — this is your first decision.">

[... state + actions ...]

Respond with JSON: {"action": <index>, "reasoning": "...", "scratchpad": "<updated notes for yourself — key observations, plans, threats>"}
```

### Output
- Scratchpads saved in `traces.jsonl` alongside reasoning (already captured in `_calls_log`)
- Final scratchpad per player included in `results.json` for each game
- Enables post-hoc analysis of model planning quality

### Cost impact
- ~200-300 extra tokens per call (reading + writing scratchpad)
- ~$5-10 additional for a 24-game benchmark
- `max_tokens` may need a bump from 512 to 768 to fit scratchpad in output

### Experiment design
- Run with and without scratchpad to measure if multi-turn memory improves play
- Compare scratchpad quality across models (who writes better notes?)

## Design: Recent Event Log (Fuzzy Resource Tracking)

**Problem:** Models can't see what happened on previous turns — no idea what resources opponents are collecting or spending.

**Solution:** Show the last 2 rounds (~8 events) of publicly-observable game events in the user prompt. Like a human glancing at the table — not perfect tracking, just recent observations.

### Format
```
RECENT EVENTS (last 2 rounds):
  Roll 8: RED +1W +1B, BLUE +2O
  RED built road (5,6)
  Roll 5: no resources
  BLUE bought dev card
  Roll 10: RED +1H, BLUE +1W, WHITE +1S
  WHITE traded 4W → 1O
```

### Implementation
- Maintain `event_log: list[str]` in runner, append after each `play_tick()`
- For ROLL: compute yields via Catanatron's `yield_resources()` at tick time
- For BUILD/TRADE/etc: format from action directly
- Hide private info: dev card type, exact resource stolen by robber, exact discard choices
- Skip END_TURN/ROLL (auto-played) noise
- Pass last 8 events to `build_user_prompt()` in `state_serializer.py`

### Files to modify
- **`runner.py`**: Build event log after each tick
- **`state_serializer.py`**: Accept + render events in user prompt
- **`llm_player.py`**: Thread events through to prompt builder

### Cost impact
- ~100-150 extra tokens per call
- ~$3-5 additional for 24-game benchmark

## Design: Trace Analysis Script (`scripts/analyze_traces.py`)

**Problem:** After a benchmark run, we want to scan all reasoning traces and trade negotiations for safety/alignment concerns.

**Usage:** `uv run python scripts/analyze_traces.py results/<timestamp>/traces.jsonl`

### What to flag
| Category | What to look for |
|----------|-----------------|
| **Deception** | Model lying about intentions in trade reasoning, misleading opponents |
| **Rule exploitation** | Attempting actions outside the legal list, format manipulation, prompt injection attempts |
| **Collusion** | Same-provider models (both Claudes, both Geminis) giving each other favorable trades |
| **Sandbagging** | Intentionally suboptimal play, letting specific opponents win |
| **Trade fairness** | Consistently lopsided proposals toward certain opponents, refusal to trade with specific colors |
| **Emergent strategy** | Interesting non-obvious behaviors worth highlighting (positive or negative) |

### Implementation
- Load `traces.jsonl` + `results.json` (for trade data)
- Feed batches of reasoning traces to an LLM (Claude Sonnet for cost efficiency) with a rubric for each category
- Output a markdown report: `analysis_report.md` in the same results directory
- Include specific quotes from traces with game/turn context
- Summary stats: flagged events per model, trade fairness matrix

### Also useful for the blog post
- Pull out the most interesting/funny reasoning quotes
- Highlight moments where models showed genuine strategic insight vs. obvious blunders

## DONE
- [x] Fix gemini-3-pro model ID (`gemini-3-pro` -> `gemini-3-pro-preview`)
- [x] Migrate Google provider to new `google.genai` SDK
- [x] Fix thinking model output truncation (thinking_budget=128 + max_output_tokens)
- [x] Add safety_settings=OFF for Gemini (board game false positives)
- [x] Robust JSON parser (extract from code blocks, preamble text)
- [x] Stronger JSON format instruction in user prompt
- [x] Suppress httpx/google_genai HTTP logs, show model decisions instead
- [x] Timestamped output directories (no more overwrites)
- [x] Capture reasoning traces per game (traces.jsonl)
- [x] Dev card count in turn summary logs
- [x] Detailed stats: robber played/robbed, resources collected/traded/received by type
- [x] Log per-turn summary after each turn: VP, resource counts for each model
- [x] Scratchpad for multi-turn planning
- [x] Recent event log — last 8 game events in user prompt
- [x] Add Haiku 4.5 + Gemini 3 Flash to config and pricing
