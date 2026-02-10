"""Game runner: orchestrates games with seat rotation and trade overlay."""

from __future__ import annotations

import itertools
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color
from catanatron.models.enums import (
    ActionType,
    RESOURCES,
    DEVELOPMENT_CARDS,
    SETTLEMENT,
    CITY,
)
from catanatron.state_functions import player_key
from catanatron.state import apply_action

from .llm_player import LLMPlayer
from .trading import TradeResult
from .metrics import GameResult, PlayerGameStats, BenchmarkMetrics

logger = logging.getLogger(__name__)

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


def generate_seat_permutations(model_names: list[str], num_games: int | None = None) -> list[list[str]]:
    """Generate seat orderings for all permutations (or limited count).

    With 4 models, there are 24 unique seat arrangements.
    """
    all_perms = list(itertools.permutations(model_names))
    if num_games is not None:
        # Cycle through permutations if num_games > len(all_perms)
        perms = []
        for i in range(num_games):
            perms.append(list(all_perms[i % len(all_perms)]))
        return perms
    return [list(p) for p in all_perms]


def run_benchmark(
    providers: dict[str, object],  # model_name -> provider instance
    num_games: int = 24,
    enable_trade: bool = True,
    seed_start: int = 42,
    max_parallel: int = 4,
    output_dir: str = "results",
) -> BenchmarkMetrics:
    """Run the full benchmark.

    Args:
        providers: Mapping of model name to provider instance
        num_games: Number of games to play
        enable_trade: Whether to enable domestic trading
        seed_start: Starting seed for reproducibility
        max_parallel: Max concurrent games (1 = sequential)
        output_dir: Directory for incremental results file

    Returns:
        BenchmarkMetrics with all results
    """
    model_names = list(providers.keys())
    seat_orders = generate_seat_permutations(model_names, num_games)
    metrics = BenchmarkMetrics()

    # Set up incremental results file
    os.makedirs(output_dir, exist_ok=True)
    incremental_path = os.path.join(output_dir, "incremental.jsonl")
    write_lock = threading.Lock()

    def _record_result(result: GameResult):
        metrics.add_game(result)
        _flush_result(result, incremental_path, write_lock)

    # Clear previous incremental file
    with open(incremental_path, "w"):
        pass

    if max_parallel <= 1:
        # Sequential execution
        for game_idx, seat_order in enumerate(seat_orders):
            seed = seed_start + game_idx
            logger.info(f"\n{'='*60}")
            logger.info(f"Game {game_idx + 1}/{num_games} | Seed: {seed}")
            logger.info(f"Seat order: {seat_order}")
            logger.info(f"{'='*60}")

            result = run_single_game(
                game_idx=game_idx,
                providers=providers,
                seat_order=seat_order,
                seed=seed,
                enable_trade=enable_trade,
            )
            _record_result(result)

            logger.info(
                f"Game {game_idx + 1} complete: winner={result.winner_model}, "
                f"turns={result.total_turns}"
            )
    else:
        # Parallel execution
        logger.info(f"Running {num_games} games with up to {max_parallel} in parallel")
        completed = 0

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {}
            for game_idx, seat_order in enumerate(seat_orders):
                seed = seed_start + game_idx
                logger.info(f"Submitting game {game_idx + 1}/{num_games} | Seed: {seed} | Seats: {seat_order}")
                future = executor.submit(
                    run_single_game,
                    game_idx=game_idx,
                    providers=providers,
                    seat_order=seat_order,
                    seed=seed,
                    enable_trade=enable_trade,
                )
                futures[future] = game_idx

            for future in as_completed(futures):
                game_idx = futures[future]
                try:
                    result = future.result()
                    _record_result(result)
                    completed += 1
                    logger.info(
                        f"Game {game_idx + 1} complete ({completed}/{num_games}): "
                        f"winner={result.winner_model}, turns={result.total_turns}"
                    )
                except Exception as e:
                    completed += 1
                    logger.error(f"Game {game_idx + 1} failed ({completed}/{num_games}): {e}")

    # Sort by game_id so reports are in order (parallel games finish out of order)
    metrics.game_results.sort(key=lambda r: r.game_id)
    metrics.compute_all()
    logger.info(f"Incremental results saved to {incremental_path}")
    return metrics


def _flush_result(result: GameResult, path: str, lock: threading.Lock):
    """Append a single GameResult to the incremental JSONL file."""
    record = {
        "game_id": result.game_id,
        "seed": result.seed,
        "winner_model": result.winner_model,
        "winner_color": result.winner_color,
        "total_turns": result.total_turns,
        "seat_order": result.seat_order,
        "trades": result.trades,
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
                "knights_played": ps.knights_played,
                "llm_calls": ps.llm_calls,
                "cost_usd": round(ps.total_cost_usd, 4),
                "input_tokens": ps.total_input_tokens,
                "output_tokens": ps.total_output_tokens,
                "cached_tokens": ps.total_cached_tokens,
                "avg_latency_ms": round(ps.avg_latency_ms, 1),
                "trade_proposals_made": ps.trade_proposals_made,
                "trade_proposals_accepted": ps.trade_proposals_accepted,
                "trade_proposals_rejected": ps.trade_proposals_rejected,
            }
            for name, ps in result.player_stats.items()
        },
        "reasoning_traces": {
            name: traces
            for name, traces in result.reasoning_traces.items()
        },
    }
    line = json.dumps(record) + "\n"
    with lock:
        with open(path, "a") as f:
            f.write(line)
            f.flush()


def run_single_game(
    game_idx: int,
    providers: dict[str, object],
    seat_order: list[str],
    seed: int,
    enable_trade: bool = True,
) -> GameResult:
    """Run a single game with the given seat order."""

    # Create players in seat order
    players = []
    for i, model_name in enumerate(seat_order):
        color = COLORS[i]
        provider = providers[model_name]
        player = LLMPlayer(color, provider, name=model_name, enable_trade=enable_trade)
        players.append(player)

    # Create game (Catanatron shuffles player order internally)
    game = Game(players, seed=seed)

    # We need to track which player object maps to which model
    # After Game.__init__, state.players might be reordered
    player_by_color = {p.color: p for p in game.state.players}

    # Give each LLM player a reference to all players (for trade targeting)
    for p in game.state.players:
        if isinstance(p, LLMPlayer):
            p._all_players_by_color = player_by_color

    trades: list[dict] = []
    event_log: list[str] = []
    actions_seen = 0  # track how many actions we've already processed
    last_trade_count = {p.color: 0 for p in game.state.players}  # track new trades

    # Custom game loop using play_tick
    while game.winning_color() is None and game.state.num_turns < TURNS_LIMIT:
        current_player = game.state.current_player()
        actions = game.state.playable_actions

        # Set recent events on LLM players before they decide
        if isinstance(current_player, LLMPlayer):
            current_player._recent_events = list(event_log)

        # Log turn summary at roll phase
        if len(actions) > 0 and actions[0].action_type == ActionType.ROLL:
            _log_turn_summary(game, player_by_color)

        # Execute one tick
        game.play_tick()

        # Record new events from actions log
        while actions_seen < len(game.state.actions):
            action = game.state.actions[actions_seen]
            actions_seen += 1
            event_str = _format_event(action, game.state.board, player_by_color)
            if event_str:
                event_log.append(event_str)

        # Collect new trade results from players and add to event log
        for p in game.state.players:
            if isinstance(p, LLMPlayer):
                new_count = len(p._trade_results)
                prev_count = last_trade_count[p.color]
                for i in range(prev_count, new_count):
                    turn, result = p._trade_results[i]
                    if result.proposal:
                        trades.append({
                            "turn": turn,
                            "proposer": result.proposal.proposer_color,
                            "target": result.proposal.target_color,
                            "accepted": result.accepted,
                            "offering": result.proposal.offering,
                            "requesting": result.proposal.requesting,
                        })
                        offering_str = ", ".join(
                            f"{v}{k}" for k, v in result.proposal.offering.items()
                            if v and int(v) > 0
                        )
                        requesting_str = ", ".join(
                            f"{v}{k}" for k, v in result.proposal.requesting.items()
                            if v and int(v) > 0
                        )
                        status = "accepted" if result.accepted else "rejected"
                        event_log.append(
                            f"{result.proposal.proposer_color} proposed trade "
                            f"({offering_str} for {requesting_str}) to "
                            f"{result.proposal.target_color}: {status}"
                        )
                last_trade_count[p.color] = new_count

    # Collect results
    winner_color = game.winning_color()
    winner_model = None
    if winner_color:
        winner_player = player_by_color.get(winner_color)
        winner_model = winner_player.name if isinstance(winner_player, LLMPlayer) else str(winner_player)

    # Gather per-player stats
    player_stats = {}
    actual_seat_order = []
    for color in game.state.colors:
        player = player_by_color[color]
        key = player_key(game.state, color)
        ps = game.state.player_state

        model_name = player.name if isinstance(player, LLMPlayer) else str(player)
        actual_seat_order.append(model_name)

        pgs = PlayerGameStats(
            model_name=model_name,
            color=color.value,
            victory_points=ps[f"{key}_ACTUAL_VICTORY_POINTS"],
            settlements=5 - ps[f"{key}_SETTLEMENTS_AVAILABLE"],
            cities=4 - ps[f"{key}_CITIES_AVAILABLE"],
            roads_built=15 - ps[f"{key}_ROADS_AVAILABLE"],
            longest_road_length=ps[f"{key}_LONGEST_ROAD_LENGTH"],
            has_longest_road=ps[f"{key}_HAS_ROAD"],
            has_largest_army=ps[f"{key}_HAS_ARMY"],
            knights_played=ps.get(f"{key}_PLAYED_KNIGHT", 0),
        )

        if isinstance(player, LLMPlayer):
            pgs.llm_calls = player.cost_tracker.total_calls
            pgs.total_cost_usd = player.cost_tracker.total_cost_usd
            pgs.total_input_tokens = player.cost_tracker.total_input_tokens
            pgs.total_output_tokens = player.cost_tracker.total_output_tokens
            pgs.total_cached_tokens = player.cost_tracker.total_cached_tokens
            pgs.avg_latency_ms = player.cost_tracker.avg_latency_ms

            # Trade stats
            for trade in trades:
                if trade.get("proposer") == color.value:
                    pgs.trade_proposals_made += 1
                    if trade.get("accepted"):
                        pgs.trade_proposals_accepted += 1
                    else:
                        pgs.trade_proposals_rejected += 1

        player_stats[model_name] = pgs

    # Scan action log for robber/resource/trade stats
    _compute_action_stats(game, player_by_color, player_stats)

    # Collect reasoning traces
    reasoning_traces = {}
    for color in game.state.colors:
        player = player_by_color[color]
        if isinstance(player, LLMPlayer):
            reasoning_traces[player.name] = list(player._calls_log)

    return GameResult(
        game_id=game_idx,
        seed=seed,
        winner_model=winner_model,
        winner_color=winner_color.value if winner_color else None,
        seat_order=actual_seat_order,
        player_stats=player_stats,
        trades=trades,
        reasoning_traces=reasoning_traces,
        total_turns=game.state.num_turns,
    )


_RES_ABBREV = {"WOOD": "W", "BRICK": "B", "SHEEP": "S", "WHEAT": "H", "ORE": "O"}
_RES_NAMES = ["W", "B", "S", "H", "O"]


def _compute_action_stats(game, player_by_color, player_stats):
    """Scan game action log to compute robber, resource, and trade stats."""
    from catanatron.state import yield_resources

    state = game.state
    board = state.board

    for action in state.actions:
        color = action.color
        player = player_by_color.get(color)
        if player is None:
            continue
        name = player.name if isinstance(player, LLMPlayer) else str(player)
        pgs = player_stats.get(name)
        if pgs is None:
            continue

        if action.action_type == ActionType.ROLL:
            # action.value is (die1, die2) in the log
            dice_val = action.value
            if isinstance(dice_val, tuple) and len(dice_val) == 2:
                dice_sum = dice_val[0] + dice_val[1]
                if dice_sum != 7:
                    # Compute who got resources from this roll
                    # yield_resources needs the bank but we just want to know production
                    # Use a full bank to avoid depletion issues
                    full_bank = [19, 19, 19, 19, 19]
                    payouts, _ = yield_resources(board, full_bank, dice_sum)
                    for payout_color, freqdeck in payouts.items():
                        p = player_by_color.get(payout_color)
                        if p is None:
                            continue
                        pname = p.name if isinstance(p, LLMPlayer) else str(p)
                        ppgs = player_stats.get(pname)
                        if ppgs is None:
                            continue
                        res_names = list(_RES_ABBREV.values())
                        for i, count in enumerate(freqdeck):
                            if count > 0:
                                ppgs.resources_collected[res_names[i]] += count

        elif action.action_type == ActionType.MOVE_ROBBER:
            # Player moved the robber
            pgs.times_played_robber += 1
            # action.value in log is (coordinate, target_color, stolen_resource)
            val = action.value
            if isinstance(val, tuple) and len(val) >= 2 and val[1] is not None:
                target = player_by_color.get(val[1])
                if target:
                    tname = target.name if isinstance(target, LLMPlayer) else str(target)
                    tpgs = player_stats.get(tname)
                    if tpgs:
                        tpgs.times_robbed += 1

        elif action.action_type == ActionType.MARITIME_TRADE:
            # value is 5-resource tuple, last is received
            val = action.value
            if isinstance(val, tuple) and len(val) == 5:
                res_list = [RESOURCES[i] if i < len(RESOURCES) else None for i in range(5)]
                # First 4 are given, last is received
                for r in val[:-1]:
                    if r is not None:
                        abbr = _RES_ABBREV.get(r)
                        if abbr:
                            pgs.resources_traded_away[abbr] += 1
                received = val[-1]
                if received is not None:
                    abbr = _RES_ABBREV.get(received)
                    if abbr:
                        pgs.resources_received[abbr] += 1


def _format_event(action, board, player_by_color) -> str | None:
    """Format a game action as a human-readable event string."""
    from catanatron.state import yield_resources

    color_name = action.color.value
    at = action.action_type
    val = action.value

    if at == ActionType.ROLL:
        if isinstance(val, tuple) and len(val) == 2:
            dice_sum = val[0] + val[1]
            if dice_sum == 7:
                return f"Roll 7: {color_name} must discard/move robber"
            # Compute payouts
            full_bank = [19, 19, 19, 19, 19]
            payouts, _ = yield_resources(board, full_bank, dice_sum)
            if not payouts:
                return f"Roll {dice_sum}: no resources"
            parts = []
            for pcolor, freqdeck in payouts.items():
                res_parts = []
                for i, count in enumerate(freqdeck):
                    if count > 0:
                        res_parts.append(f"+{count}{_RES_NAMES[i]}")
                if res_parts:
                    parts.append(f"{pcolor.value} {''.join(res_parts)}")
            return f"Roll {dice_sum}: {', '.join(parts)}" if parts else f"Roll {dice_sum}: no resources"
        return None

    if at == ActionType.BUILD_SETTLEMENT:
        return f"{color_name} built settlement node={val}"
    if at == ActionType.BUILD_CITY:
        return f"{color_name} built city node={val}"
    if at == ActionType.BUILD_ROAD:
        return f"{color_name} built road edge={val}"
    if at == ActionType.BUY_DEVELOPMENT_CARD:
        return f"{color_name} bought dev card"
    if at == ActionType.MARITIME_TRADE:
        if isinstance(val, tuple) and len(val) == 5:
            giving = [_RES_ABBREV.get(r, "?") for r in val[:-1] if r is not None]
            receiving = _RES_ABBREV.get(val[-1], "?")
            return f"{color_name} traded {''.join(giving)} -> 1{receiving}"
        return f"{color_name} maritime trade"
    if at == ActionType.MOVE_ROBBER:
        if isinstance(val, tuple) and len(val) >= 2:
            target = val[1].value if val[1] else "nobody"
            return f"{color_name} moved robber, stole from {target}"
        return f"{color_name} moved robber"
    if at == ActionType.DISCARD:
        if val:
            count = len(val) if isinstance(val, (list, tuple)) else 1
            return f"{color_name} discarded {count} cards"
        return f"{color_name} discarded"
    if at == ActionType.PLAY_KNIGHT_CARD:
        return f"{color_name} played Knight"
    if at == ActionType.PLAY_MONOPOLY:
        res = _RES_ABBREV.get(val, str(val))
        return f"{color_name} played Monopoly on {res}"
    if at == ActionType.PLAY_YEAR_OF_PLENTY:
        return f"{color_name} played Year of Plenty"
    if at == ActionType.PLAY_ROAD_BUILDING:
        return f"{color_name} played Road Building"
    if at == ActionType.END_TURN:
        return None

    return None


def _log_turn_summary(game, player_by_color):
    """Log a one-line-per-player summary at the start of each turn."""
    state = game.state
    turn = state.num_turns
    current = state.current_color()
    parts = []
    for color in state.colors:
        key = player_key(state, color)
        ps = state.player_state
        vp = ps[f"{key}_ACTUAL_VICTORY_POINTS"]
        res = "".join(
            f"{abbr}{ps[f'{key}_{full}_IN_HAND']}"
            for full, abbr in _RES_ABBREV.items()
            if ps[f"{key}_{full}_IN_HAND"] > 0
        )
        dev_count = sum(ps[f"{key}_{dev}_IN_HAND"] for dev in DEVELOPMENT_CARDS)
        dev_str = f" dev={dev_count}" if dev_count > 0 else ""
        player = player_by_color[color]
        name = player.name if isinstance(player, LLMPlayer) else str(player)
        marker = "*" if color == current else " "
        parts.append(f"{marker}{name}(VP={vp} {res or '-'}{dev_str})")
    logger.info(f"Turn {turn}: {' | '.join(parts)}")
