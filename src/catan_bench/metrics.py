"""Game results, model statistics, ELO ratings, and Wilson confidence intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class GameResult:
    """Result of a single game."""
    game_id: int
    seed: int
    winner_model: str | None  # None if game timed out
    winner_color: str | None
    seat_order: list[str]  # model names in seat order
    player_stats: dict[str, PlayerGameStats]  # model_name -> stats
    trades: list[dict] = field(default_factory=list)
    reasoning_traces: dict[str, list[dict]] = field(default_factory=dict)  # model_name -> calls_log
    total_turns: int = 0


@dataclass
class PlayerGameStats:
    """Per-player stats for a single game."""
    model_name: str
    color: str
    victory_points: int = 0
    settlements: int = 0
    cities: int = 0
    roads_built: int = 0
    longest_road_length: int = 0
    has_longest_road: bool = False
    has_largest_army: bool = False
    knights_played: int = 0
    dev_cards_bought: int = 0
    total_resources_collected: int = 0
    llm_calls: int = 0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    avg_latency_ms: float = 0.0
    trade_proposals_made: int = 0
    trade_proposals_accepted: int = 0
    trade_proposals_rejected: int = 0
    # Robber stats
    times_robbed: int = 0
    times_played_robber: int = 0
    # Resource stats by type: {resource_abbrev: count}
    resources_collected: dict[str, int] = field(default_factory=lambda: {"W": 0, "B": 0, "S": 0, "H": 0, "O": 0})
    resources_traded_away: dict[str, int] = field(default_factory=lambda: {"W": 0, "B": 0, "S": 0, "H": 0, "O": 0})
    resources_received: dict[str, int] = field(default_factory=lambda: {"W": 0, "B": 0, "S": 0, "H": 0, "O": 0})


@dataclass
class ModelStats:
    """Aggregated statistics across all games for a model."""
    model_name: str
    games_played: int = 0
    wins: int = 0
    win_rate: float = 0.0
    win_rate_ci_low: float = 0.0
    win_rate_ci_high: float = 0.0
    elo_rating: float = 1500.0
    avg_victory_points: float = 0.0
    avg_settlements: float = 0.0
    avg_cities: float = 0.0
    avg_roads: float = 0.0
    longest_road_count: int = 0
    largest_army_count: int = 0
    total_cost_usd: float = 0.0
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    avg_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    cost_per_game: float = 0.0
    # Trade stats
    trade_proposals_made: int = 0
    trade_acceptance_rate: float = 0.0
    # Robber stats
    avg_times_robbed: float = 0.0
    avg_times_played_robber: float = 0.0
    # Resource stats (averages per game)
    avg_resources_collected: dict[str, float] = field(default_factory=lambda: {"W": 0, "B": 0, "S": 0, "H": 0, "O": 0})
    avg_resources_traded_away: dict[str, float] = field(default_factory=lambda: {"W": 0, "B": 0, "S": 0, "H": 0, "O": 0})
    avg_resources_received: dict[str, float] = field(default_factory=lambda: {"W": 0, "B": 0, "S": 0, "H": 0, "O": 0})


@dataclass
class BenchmarkMetrics:
    """Full benchmark results."""
    game_results: list[GameResult] = field(default_factory=list)
    model_stats: dict[str, ModelStats] = field(default_factory=dict)
    total_games: int = 0
    total_cost_usd: float = 0.0

    def add_game(self, result: GameResult):
        self.game_results.append(result)
        self.total_games += 1

    def compute_all(self):
        """Recompute all aggregate stats from game results."""
        self.model_stats.clear()
        self.total_cost_usd = 0.0

        # Accumulate per-model stats
        for result in self.game_results:
            for model_name, pstats in result.player_stats.items():
                if model_name not in self.model_stats:
                    self.model_stats[model_name] = ModelStats(model_name=model_name)
                ms = self.model_stats[model_name]
                ms.games_played += 1

                if result.winner_model == model_name:
                    ms.wins += 1

                ms.avg_victory_points += pstats.victory_points
                ms.avg_settlements += pstats.settlements
                ms.avg_cities += pstats.cities
                ms.avg_roads += pstats.roads_built
                if pstats.has_longest_road:
                    ms.longest_road_count += 1
                if pstats.has_largest_army:
                    ms.largest_army_count += 1

                ms.total_cost_usd += pstats.total_cost_usd
                ms.total_llm_calls += pstats.llm_calls
                ms.total_input_tokens += pstats.total_input_tokens
                ms.total_output_tokens += pstats.total_output_tokens
                ms.total_cached_tokens += pstats.total_cached_tokens
                ms.avg_latency_ms += pstats.avg_latency_ms * pstats.llm_calls

                ms.trade_proposals_made += pstats.trade_proposals_made
                trade_total = pstats.trade_proposals_accepted + pstats.trade_proposals_rejected
                if trade_total > 0:
                    ms.trade_acceptance_rate += pstats.trade_proposals_accepted

                ms.avg_times_robbed += pstats.times_robbed
                ms.avg_times_played_robber += pstats.times_played_robber
                for res in ("W", "B", "S", "H", "O"):
                    ms.avg_resources_collected[res] += pstats.resources_collected[res]
                    ms.avg_resources_traded_away[res] += pstats.resources_traded_away[res]
                    ms.avg_resources_received[res] += pstats.resources_received[res]

        # Compute averages and derived metrics
        for ms in self.model_stats.values():
            n = ms.games_played
            if n > 0:
                ms.win_rate = ms.wins / n
                ms.avg_victory_points /= n
                ms.avg_settlements /= n
                ms.avg_cities /= n
                ms.avg_roads /= n
                ms.cost_per_game = ms.total_cost_usd / n

                if ms.total_llm_calls > 0:
                    ms.avg_latency_ms /= ms.total_llm_calls

                total_input = ms.total_input_tokens + ms.total_cached_tokens
                if total_input > 0:
                    ms.cache_hit_rate = ms.total_cached_tokens / total_input

                if ms.trade_proposals_made > 0:
                    ms.trade_acceptance_rate /= ms.trade_proposals_made

                ms.avg_times_robbed /= n
                ms.avg_times_played_robber /= n
                for res in ("W", "B", "S", "H", "O"):
                    ms.avg_resources_collected[res] /= n
                    ms.avg_resources_traded_away[res] /= n
                    ms.avg_resources_received[res] /= n

                # Wilson score CI
                ms.win_rate_ci_low, ms.win_rate_ci_high = wilson_score_ci(ms.wins, n)

            self.total_cost_usd += ms.total_cost_usd

        # ELO ratings
        self._compute_elo()

    def _compute_elo(self):
        """Compute multi-player ELO from game results."""
        # Initialize
        elo = {name: 1500.0 for name in self.model_stats}
        K = 32.0

        for result in self.game_results:
            players = list(result.player_stats.keys())
            n = len(players)
            if n < 2:
                continue

            # For each pair, update ELO based on relative performance
            for i in range(n):
                for j in range(i + 1, n):
                    pi, pj = players[i], players[j]
                    # Expected scores
                    ei = 1.0 / (1.0 + 10 ** ((elo[pj] - elo[pi]) / 400))
                    ej = 1.0 - ei

                    # Actual scores: winner gets 1, loser gets 0, tie 0.5
                    if result.winner_model == pi:
                        si, sj = 1.0, 0.0
                    elif result.winner_model == pj:
                        si, sj = 0.0, 1.0
                    else:
                        si, sj = 0.5, 0.5  # timeout = draw

                    # K factor scaled for multi-player (each pair is 1/(n-1) of a game)
                    k_scaled = K / (n - 1)
                    elo[pi] += k_scaled * (si - ei)
                    elo[pj] += k_scaled * (sj - ej)

        for name, rating in elo.items():
            if name in self.model_stats:
                self.model_stats[name].elo_rating = round(rating, 1)


def wilson_score_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for binomial proportion.

    Args:
        wins: Number of successes
        n: Total trials
        z: Z-score (1.96 for 95% CI)

    Returns:
        (lower, upper) bounds
    """
    if n == 0:
        return 0.0, 0.0

    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom

    return max(0.0, round(center - spread, 4)), min(1.0, round(center + spread, 4))
