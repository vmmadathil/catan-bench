# Catan Bench

Benchmark frontier LLMs against each other in Settlers of Catan.

Four models sit at a table. Each sees a text description of the board, their hand, recent events, and a numbered list of legal moves. They pick one, explain why, and write notes to themselves for next turn. No fine-tuning, no search, no RL. Just prompting.

The goal is to test how LLMs handle multi-agent strategic reasoning: resource management, probabilistic planning, negotiation, and adapting to opponents over long horizons. The kind of messy, multi-stakeholder problem that's hard to capture with multiple choice benchmarks.

## Quickstart

```bash
# Clone and install
git clone https://github.com/your-username/catan-bench.git
cd catan-bench
uv sync

# Set API keys
export ANTHROPIC_API_KEY=sk-...
export GOOGLE_API_KEY=...

# Run a single test game
uv run python scripts/run_benchmark.py -g 1 --models gemini-flash claude-sonnet claude-haiku gemini-3-flash

# Run a full 48-game tournament
uv run python scripts/run_benchmark.py -g 48 --models gemini-flash claude-sonnet claude-haiku gemini-3-flash
```

## How it works

Each game runs on [Catanatron](https://github.com/bcollazo/catanatron), an open-source Catan engine. The models never touch the engine directly. They receive:

- A system prompt with the full rules of Catan and the static board layout (cached for cost efficiency)
- The current game state: buildings, roads, VP counts, robber position
- Their hand (opponents' hands are hidden, just like real Catan)
- The last 8 game events
- A scratchpad with their notes from the previous turn
- A numbered list of legal actions

They respond with JSON: an action index, brief reasoning, and updated scratchpad notes.

Domestic trading is bolted on top of Catanatron's engine. After each dice roll, the active player can propose a 1-for-1 trade to any opponent, who independently accepts or rejects.

## CLI options

```
uv run python scripts/run_benchmark.py [OPTIONS]

--config, -c     Path to config YAML (default: config/default.yaml)
--games, -g      Number of games (default: 24)
--models, -m     Models to include (space-separated names from config)
--no-trade       Disable domestic trading
--output, -o     Output directory
--seed           Starting random seed
--parallel, -p   Max concurrent games (default: 4)
```

## Available models

Configured in `config/default.yaml`:

| Name | Provider | Model ID |
|------|----------|----------|
| `claude-sonnet` | Anthropic | `claude-sonnet-4-5-20250929` |
| `claude-haiku` | Anthropic | `claude-haiku-4-5-20251001` |
| `claude-opus` | Anthropic | `claude-opus-4-6` |
| `gemini-flash` | Google | `gemini-2.5-flash` |
| `gemini-3-flash` | Google | `gemini-3-flash-preview` |
| `gemini-3-pro` | Google | `gemini-3-pro-preview` |
| `gemini-2.5-pro` | Google | `gemini-2.5-pro` |

## Output

Each run creates a timestamped directory under `results/` containing:

- `summary.txt` — leaderboard, per-model stats, game-by-game results
- `results.json` — structured results for programmatic analysis
- `traces.jsonl` — every LLM call: prompt, response, reasoning, scratchpad, tokens, latency
- Per-game logs in `games/`

## Generating charts

```bash
uv run python scripts/generate_charts.py
```

Outputs to `charts/`.

## Project structure

```
catan-bench/
├── config/default.yaml          # Model configs and benchmark settings
├── scripts/
│   ├── run_benchmark.py         # CLI entry point
│   └── generate_charts.py       # Blog chart generation
├── src/catan_bench/
│   ├── runner.py                # Game loop and orchestration
│   ├── llm_player.py            # LLM-as-Catan-player wrapper
│   ├── state_serializer.py      # Board/state → text prompt
│   ├── trading.py               # Domestic trade overlay
│   ├── analysis.py              # Post-run report generation
│   ├── metrics.py               # ELO, win rates, confidence intervals
│   └── providers/
│       ├── anthropic.py         # Claude API
│       ├── google.py            # Gemini API
│       └── openai.py            # OpenAI API (stub)
├── results/                     # Timestamped benchmark outputs
├── charts/                      # Generated visualizations
└── blog-draft.md                # Write-up of findings
```

## Cost

A 48-game tournament with 4 models costs ~$25 and takes ~8 hours, bottlenecked by API throughput.
