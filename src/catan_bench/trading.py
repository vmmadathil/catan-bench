"""Inter-player trade negotiation overlay on Catanatron.

Catanatron only supports maritime trade. This module provides domestic trade
support integrated into the LLMPlayer.decide() flow:
- The proposer's LLM sees a "DOMESTIC TRADE OPTION" in the main decide() prompt
- The responder gets a separate LLM call with full cached context
- On accept: mutate game.state.player_state to swap resources
- 1 trade proposal per turn
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from catanatron.models.enums import RESOURCES
from catanatron.state_functions import player_key

from .state_serializer import RESOURCE_ABBREV, RESOURCE_FULL

logger = logging.getLogger(__name__)


@dataclass
class TradeProposal:
    """A trade proposal from one player to another."""
    proposer_color: str
    target_color: str
    offering: dict[str, int]  # resource_abbrev -> count
    requesting: dict[str, int]  # resource_abbrev -> count


@dataclass
class TradeResult:
    """Result of a trade negotiation."""
    proposal: TradeProposal | None
    accepted: bool
    proposer_reasoning: str = ""
    responder_reasoning: str = ""


@dataclass
class TradeStats:
    """Accumulated trade statistics for a model."""
    proposals_made: int = 0
    proposals_received: int = 0
    proposals_accepted_as_proposer: int = 0
    proposals_accepted_as_responder: int = 0
    proposals_rejected_as_proposer: int = 0
    proposals_rejected_as_responder: int = 0
    passed_on_trade: int = 0


def build_responder_prompt(
    game,
    responder_color,
    proposer_color_str: str,
    offering: dict[str, int],
    requesting: dict[str, int],
    scratchpad: str = "",
    recent_events: list[str] | None = None,
) -> str:
    """Build a user prompt for the trade responder with full game context.

    The responder's cached system prompt provides board layout and rules.
    This user prompt adds dynamic state + trade details.
    """
    from .state_serializer import serialize_dynamic_state

    dynamic = serialize_dynamic_state(game, responder_color)
    sections = [dynamic]

    # Recent events (last 8)
    if recent_events:
        events = recent_events[-8:]
        sections.append("RECENT EVENTS:\n" + "\n".join(f"  {e}" for e in events))

    # Scratchpad
    notes = scratchpad if scratchpad else "No notes yet."
    sections.append(f"YOUR NOTES (from previous turns):\n{notes}")

    # Trade details
    offering_str = ", ".join(f"{v} {k}" for k, v in offering.items() if v and int(v) > 0)
    requesting_str = ", ".join(f"{v} {k}" for k, v in requesting.items() if v and int(v) > 0)

    sections.append(
        f"TRADE PROPOSAL FROM {proposer_color_str}:\n"
        f"  They offer you: {offering_str}\n"
        f"  They want from you: {requesting_str}\n"
        f"Consider whether this trade helps your position more than theirs.\n"
        f'Respond with ONLY a JSON object:\n'
        f'  To accept: {{"action": "accept", "reasoning": "..."}}\n'
        f'  To reject: {{"action": "reject", "reasoning": "..."}}'
    )

    return "\n\n".join(sections)


def get_opponent_info(game, player_color) -> list[dict]:
    """Get info about opponents for trade targeting."""
    state = game.state
    opponents = []
    for color in state.colors:
        if color == player_color:
            continue
        key = player_key(state, color)
        total = sum(state.player_state[f"{key}_{res}_IN_HAND"] for res in RESOURCES)
        opponents.append({
            "color": color,
            "color_name": color.value,
            "total_cards": total,
        })
    return opponents


def format_player_resources(state, color) -> str:
    """Format a player's resources as a compact string."""
    key = player_key(state, color)
    parts = []
    for res in RESOURCES:
        count = state.player_state[f"{key}_{res}_IN_HAND"]
        if count > 0:
            parts.append(f"{RESOURCE_ABBREV[res]}={count}")
    return ", ".join(parts) if parts else "empty"


def validate_resources(state, color, resource_dict: dict) -> bool:
    """Check if a player has the specified resources."""
    key = player_key(state, color)
    for abbrev, count in resource_dict.items():
        try:
            count = int(count)
        except (ValueError, TypeError):
            return False
        if count <= 0:
            continue
        full_name = RESOURCE_FULL.get(abbrev)
        if full_name is None:
            return False
        available = state.player_state[f"{key}_{full_name}_IN_HAND"]
        if available < count:
            return False
    return True


def execute_trade(state, proposer_color, responder_color, offering: dict, requesting: dict):
    """Mutate game state to execute a trade."""
    p_key = player_key(state, proposer_color)
    r_key = player_key(state, responder_color)

    # Move offered resources: proposer -> responder
    for abbrev, count in offering.items():
        count = int(count)
        if count <= 0:
            continue
        full_name = RESOURCE_FULL[abbrev]
        state.player_state[f"{p_key}_{full_name}_IN_HAND"] -= count
        state.player_state[f"{r_key}_{full_name}_IN_HAND"] += count

    # Move requested resources: responder -> proposer
    for abbrev, count in requesting.items():
        count = int(count)
        if count <= 0:
            continue
        full_name = RESOURCE_FULL[abbrev]
        state.player_state[f"{r_key}_{full_name}_IN_HAND"] -= count
        state.player_state[f"{p_key}_{full_name}_IN_HAND"] += count
