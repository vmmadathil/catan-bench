"""Report generation: CSV, JSON, and human-readable summary."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from .metrics import BenchmarkMetrics, GameResult


def generate_reports(metrics: BenchmarkMetrics, output_dir: str = "results") -> dict[str, str]:
    """Generate all report files.

    Returns:
        Dict of report_name -> file_path
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    paths = {}
    paths["summary"] = _write_summary(metrics, run_dir, timestamp)
    paths["csv"] = _write_csv(metrics, run_dir, timestamp)
    paths["json"] = _write_json(metrics, run_dir, timestamp)
    paths["traces"] = _write_traces(metrics, run_dir, timestamp)

    return paths


def _write_summary(metrics: BenchmarkMetrics, output_dir: str, timestamp: str) -> str:
    """Write human-readable summary."""
    path = os.path.join(output_dir, "summary.txt")
    lines = []

    lines.append("=" * 70)
    lines.append("CATAN LLM BENCHMARK RESULTS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total games: {metrics.total_games}")
    lines.append(f"Total cost: ${metrics.total_cost_usd:.2f}")
    lines.append("=" * 70)

    # Leaderboard
    lines.append("\nLEADERBOARD (by ELO):")
    lines.append("-" * 70)
    lines.append(f"{'Model':<30} {'ELO':>6} {'Wins':>5} {'WR%':>6} {'95% CI':>14} {'Avg VP':>7}")
    lines.append("-" * 70)

    sorted_models = sorted(
        metrics.model_stats.values(),
        key=lambda m: m.elo_rating,
        reverse=True,
    )

    for ms in sorted_models:
        ci = f"[{ms.win_rate_ci_low:.1%}-{ms.win_rate_ci_high:.1%}]"
        lines.append(
            f"{ms.model_name:<30} {ms.elo_rating:>6.1f} {ms.wins:>5} "
            f"{ms.win_rate:>5.1%} {ci:>14} {ms.avg_victory_points:>7.1f}"
        )

    # Detailed stats
    lines.append("\n\nDETAILED STATISTICS:")
    lines.append("-" * 70)
    for ms in sorted_models:
        lines.append(f"\n{ms.model_name}:")
        lines.append(f"  Games: {ms.games_played} | Wins: {ms.wins} | Win rate: {ms.win_rate:.1%}")
        lines.append(f"  ELO: {ms.elo_rating:.1f}")
        lines.append(f"  Avg VP: {ms.avg_victory_points:.1f} | Avg settlements: {ms.avg_settlements:.1f} | Avg cities: {ms.avg_cities:.1f}")
        lines.append(f"  Longest road: {ms.longest_road_count}x | Largest army: {ms.largest_army_count}x")
        lines.append(f"  Cost: ${ms.total_cost_usd:.2f} total, ${ms.cost_per_game:.2f}/game")
        lines.append(f"  LLM calls: {ms.total_llm_calls} | Avg latency: {ms.avg_latency_ms:.0f}ms")
        lines.append(f"  Tokens: {ms.total_input_tokens:,} input, {ms.total_output_tokens:,} output, {ms.total_cached_tokens:,} cached")
        lines.append(f"  Cache hit rate: {ms.cache_hit_rate:.1%}")
        if ms.trade_proposals_made > 0:
            lines.append(f"  Trades proposed: {ms.trade_proposals_made} | Acceptance rate: {ms.trade_acceptance_rate:.1%}")
        lines.append(f"  Robber: played {ms.avg_times_played_robber:.1f}x/game | robbed {ms.avg_times_robbed:.1f}x/game")
        collected = " ".join(f"{r}={v:.1f}" for r, v in ms.avg_resources_collected.items())
        lines.append(f"  Avg resources collected/game: {collected}")
        traded = " ".join(f"{r}={v:.1f}" for r, v in ms.avg_resources_traded_away.items())
        received = " ".join(f"{r}={v:.1f}" for r, v in ms.avg_resources_received.items())
        lines.append(f"  Avg traded away/game: {traded}")
        lines.append(f"  Avg received/game: {received}")

    # Game-by-game results
    lines.append("\n\nGAME-BY-GAME RESULTS:")
    lines.append("-" * 70)
    lines.append(f"{'#':>3} {'Winner':<25} {'VP':>3} {'Turns':>6} {'Seat Order'}")
    lines.append("-" * 70)
    for r in metrics.game_results:
        winner = r.winner_model or "TIMEOUT"
        winner_vp = ""
        if r.winner_model and r.winner_model in r.player_stats:
            winner_vp = str(r.player_stats[r.winner_model].victory_points)
        lines.append(
            f"{r.game_id + 1:>3} {winner:<25} {winner_vp:>3} {r.total_turns:>6} {' -> '.join(r.seat_order)}"
        )

    content = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(content)
    return path


def _write_csv(metrics: BenchmarkMetrics, output_dir: str, timestamp: str) -> str:
    """Write per-game CSV."""
    path = os.path.join(output_dir, "games.csv")

    fieldnames = [
        "game_id", "seed", "winner_model", "winner_color", "total_turns",
        "model", "color", "seat_position", "victory_points", "settlements",
        "cities", "roads_built", "longest_road_length", "has_longest_road",
        "has_largest_army", "knights_played", "llm_calls", "cost_usd",
        "input_tokens", "output_tokens", "cached_tokens", "avg_latency_ms",
        "trade_proposals_made", "trade_proposals_accepted",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in metrics.game_results:
            for seat_pos, model_name in enumerate(result.seat_order):
                pstats = result.player_stats.get(model_name)
                if not pstats:
                    continue
                writer.writerow({
                    "game_id": result.game_id,
                    "seed": result.seed,
                    "winner_model": result.winner_model or "TIMEOUT",
                    "winner_color": result.winner_color or "",
                    "total_turns": result.total_turns,
                    "model": model_name,
                    "color": pstats.color,
                    "seat_position": seat_pos,
                    "victory_points": pstats.victory_points,
                    "settlements": pstats.settlements,
                    "cities": pstats.cities,
                    "roads_built": pstats.roads_built,
                    "longest_road_length": pstats.longest_road_length,
                    "has_longest_road": pstats.has_longest_road,
                    "has_largest_army": pstats.has_largest_army,
                    "knights_played": pstats.knights_played,
                    "llm_calls": pstats.llm_calls,
                    "cost_usd": round(pstats.total_cost_usd, 4),
                    "input_tokens": pstats.total_input_tokens,
                    "output_tokens": pstats.total_output_tokens,
                    "cached_tokens": pstats.total_cached_tokens,
                    "avg_latency_ms": round(pstats.avg_latency_ms, 1),
                    "trade_proposals_made": pstats.trade_proposals_made,
                    "trade_proposals_accepted": pstats.trade_proposals_accepted,
                })

    return path


def _write_json(metrics: BenchmarkMetrics, output_dir: str, timestamp: str) -> str:
    """Write full results as JSON."""
    path = os.path.join(output_dir, "results.json")

    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_games": metrics.total_games,
            "total_cost_usd": round(metrics.total_cost_usd, 4),
        },
        "leaderboard": [
            {
                "model": ms.model_name,
                "elo": ms.elo_rating,
                "wins": ms.wins,
                "games": ms.games_played,
                "win_rate": round(ms.win_rate, 4),
                "win_rate_ci": [ms.win_rate_ci_low, ms.win_rate_ci_high],
                "avg_victory_points": round(ms.avg_victory_points, 2),
                "avg_settlements": round(ms.avg_settlements, 2),
                "avg_cities": round(ms.avg_cities, 2),
                "avg_roads": round(ms.avg_roads, 2),
                "longest_road_count": ms.longest_road_count,
                "largest_army_count": ms.largest_army_count,
                "total_cost_usd": round(ms.total_cost_usd, 4),
                "cost_per_game": round(ms.cost_per_game, 4),
                "total_llm_calls": ms.total_llm_calls,
                "cache_hit_rate": round(ms.cache_hit_rate, 4),
                "avg_latency_ms": round(ms.avg_latency_ms, 1),
                "trade_proposals_made": ms.trade_proposals_made,
                "trade_acceptance_rate": round(ms.trade_acceptance_rate, 4),
            }
            for ms in sorted(
                metrics.model_stats.values(),
                key=lambda m: m.elo_rating,
                reverse=True,
            )
        ],
        "games": [
            {
                "game_id": r.game_id,
                "seed": r.seed,
                "winner_model": r.winner_model,
                "winner_color": r.winner_color,
                "total_turns": r.total_turns,
                "seat_order": r.seat_order,
                "trades": r.trades,
                "players": {
                    name: {
                        "color": ps.color,
                        "victory_points": ps.victory_points,
                        "settlements": ps.settlements,
                        "cities": ps.cities,
                        "roads_built": ps.roads_built,
                        "longest_road_length": ps.longest_road_length,
                        "has_longest_road": ps.has_longest_road,
                        "has_largest_army": ps.has_largest_army,
                        "llm_calls": ps.llm_calls,
                        "cost_usd": round(ps.total_cost_usd, 4),
                    }
                    for name, ps in r.player_stats.items()
                },
            }
            for r in metrics.game_results
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def _write_traces(metrics: BenchmarkMetrics, output_dir: str, timestamp: str) -> str:
    """Write per-game reasoning traces."""
    path = os.path.join(output_dir, "traces.jsonl")

    with open(path, "w") as f:
        for result in metrics.game_results:
            record = {
                "game_id": result.game_id,
                "seed": result.seed,
                "winner": result.winner_model,
                "total_turns": result.total_turns,
                "traces": {},
            }
            for model_name, trace in result.reasoning_traces.items():
                record["traces"][model_name] = trace
            f.write(json.dumps(record) + "\n")

    return path
