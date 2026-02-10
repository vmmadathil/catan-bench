"""Serialize Catanatron game state into compact text for LLM prompts.

Board layout (static) goes in the system prompt for caching.
Dynamic state uses abbreviated resources: W/B/S/H/O.
Actions are indexed for easy LLM selection.
"""

from __future__ import annotations

from catanatron.models.enums import (
    ActionType,
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
    SETTLEMENT,
    CITY,
    RESOURCES,
    DEVELOPMENT_CARDS,
)
from catanatron.models.map import LandTile, NodeRef
from catanatron.state_functions import player_key

RESOURCE_ABBREV = {
    WOOD: "W",
    BRICK: "B",
    SHEEP: "S",
    WHEAT: "H",  # wHeat to avoid clash with Wood
    ORE: "O",
}

RESOURCE_FULL = {v: k for k, v in RESOURCE_ABBREV.items()}


def serialize_board(game) -> str:
    """Serialize the static board layout for the system prompt (cached).

    Includes: tile resources/numbers, node-tile adjacency, port locations.
    """
    state = game.state
    board = state.board
    catan_map = board.map
    lines = []

    lines.append("=== BOARD LAYOUT ===")
    lines.append("Tiles (id: resource number, nodes):")
    for coord, tile in sorted(catan_map.land_tiles.items(), key=lambda x: x[1].id):
        if tile.resource is None:
            res_str = "DESERT"
            num_str = ""
        else:
            res_str = RESOURCE_ABBREV.get(tile.resource, tile.resource)
            num_str = f" #{tile.number}"
        node_ids = sorted(tile.nodes.values())
        lines.append(f"  T{tile.id}: {res_str}{num_str} nodes={node_ids}")

    # Ports
    lines.append("Ports:")
    for resource, node_ids in catan_map.port_nodes.items():
        if resource is None:
            lines.append(f"  3:1 port at nodes {sorted(node_ids)}")
        else:
            abbrev = RESOURCE_ABBREV.get(resource, resource)
            lines.append(f"  2:1 {abbrev} port at nodes {sorted(node_ids)}")

    # Robber starting position
    lines.append(f"Robber starts at: {board.robber_coordinate}")

    return "\n".join(lines)


def serialize_system_prompt(game, player_color) -> str:
    """Full system prompt including rules, board layout, and player identity."""
    board_layout = serialize_board(game)
    color_name = player_color.value

    return f"""You are an expert Settlers of Catan player. You are playing as {color_name}.

GAME RULES:
Victory: First player to reach 10 VP on their turn wins immediately.
VP sources: settlement=1, city=2, longest road (5+ segments)=2, largest army (3+ knights)=2, VP dev cards=1 each.

Resources: W=Wood, B=Brick, S=Sheep, H=Wheat, O=Ore
Build costs: Road=W+B, Settlement=W+B+S+H, City=3O+2H, Dev card=S+H+O

Dice & Production:
- Each turn starts with a dice roll (2d6). Every tile matching the number produces resources.
- Settlements on a producing tile get 1 resource; cities get 2.
- Tile probability (pips): 2→1, 3→2, 4→3, 5→4, 6→5, 8→5, 9→4, 10→3, 11→2, 12→1. Higher pips = more frequent. 6 and 8 are best; 2 and 12 are worst.

Rolling a 7 / Robber:
- Any player with more than 7 resource cards must discard half (rounded down).
- The active player moves the robber to any tile, blocking its production.
- The active player steals 1 random resource from any player with a building on that tile.

Building Rules:
- Settlements must be at least 2 edges away from any other settlement or city.
- Roads must connect to your existing road network or buildings.
- Cities replace (upgrade) your existing settlements.

Development Cards:
- Knight: Move the robber and steal (same as rolling 7). Counts toward Largest Army.
- Year of Plenty: Take any 2 resources from the bank.
- Monopoly: Name 1 resource; ALL other players give you ALL of that resource from their hand.
- Road Building: Build 2 roads for free.
- Victory Point: Kept hidden in your hand. These automatically count toward your VP total — you do NOT need to play them. They are never revealed until you win.
- You may play at most 1 dev card per turn. You cannot play a card the same turn you bought it.

Longest Road: Awarded to the first player with 5+ connected road segments. Stolen if another player builds a longer continuous road. 
Largest Army: Awarded to the first player with 3+ knights played. Stolen if another player plays more knights. 

Trading:
- Maritime trade: 4:1 (any 4 of same resource for 1 of any), 3:1 with generic port, 2:1 with specific resource port.
- Domestic trade: you may propose a trade with one other player per turn (when available).

{board_layout}

RESPONSE FORMAT:
You must respond with valid JSON: {{"action": <index>, "reasoning": "<brief explanation>", "scratchpad": "<your updated notes for future turns>"}}
The "action" value must be the [index] number shown next to your chosen action. For example, if actions are [0] BUILD_ROAD, [1] END_TURN and you want to build a road, respond with {{"action": 0, ...}}.
Use the scratchpad to record your strategic notes — it will be shown back to you on future turns. Keep it concise (max 2000 characters).

CONSTRAINT: Your response is limited to 1536 tokens. Be concise — prioritize your action choice and key strategic notes over lengthy reasoning."""


def serialize_dynamic_state(game, player_color) -> str:
    """Serialize the dynamic game state for the user prompt (changes each turn)."""
    state = game.state
    lines = []

    lines.append(f"=== TURN {state.num_turns} ===")
    lines.append(f"Current player: {state.current_color().value}")

    # All players' visible state
    lines.append("\nPLAYER STATUS:")
    for color in state.colors:
        key = player_key(state, color)
        ps = state.player_state
        is_you = " (YOU)" if color == player_color else ""

        vp = ps[f"{key}_VICTORY_POINTS"]
        settlements = 5 - ps[f"{key}_SETTLEMENTS_AVAILABLE"]
        cities = 4 - ps[f"{key}_CITIES_AVAILABLE"]
        roads = 15 - ps[f"{key}_ROADS_AVAILABLE"]
        has_road = ps[f"{key}_HAS_ROAD"]
        has_army = ps[f"{key}_HAS_ARMY"]
        longest_road_len = ps[f"{key}_LONGEST_ROAD_LENGTH"]
        knights_played = ps.get(f"{key}_PLAYED_KNIGHT", 0)

        line = f"  {color.value}{is_you}: VP={vp} settlements={settlements} cities={cities} roads={roads}"
        line += f" longest_road={longest_road_len}"
        if has_road:
            line += " [LONGEST ROAD]"
        if has_army:
            line += " [LARGEST ARMY]"

        # Show total cards for opponents, exact hand for self
        if color == player_color:
            hand_parts = []
            for res in RESOURCES:
                count = ps[f"{key}_{res}_IN_HAND"]
                if count > 0:
                    hand_parts.append(f"{RESOURCE_ABBREV[res]}={count}")
            hand_str = ",".join(hand_parts) if hand_parts else "empty"
            line += f" hand=[{hand_str}]"

            # Dev cards in hand
            dev_parts = []
            for dev in DEVELOPMENT_CARDS:
                count = ps[f"{key}_{dev}_IN_HAND"]
                if count > 0:
                    dev_parts.append(f"{dev}={count}")
            if dev_parts:
                line += f" dev_cards=[{','.join(dev_parts)}]"
        else:
            total_resources = sum(ps[f"{key}_{res}_IN_HAND"] for res in RESOURCES)
            total_dev = sum(ps[f"{key}_{dev}_IN_HAND"] for dev in DEVELOPMENT_CARDS)
            line += f" cards={total_resources} dev_cards={total_dev}"

        lines.append(line)

    # Buildings on the board
    lines.append("\nBUILDINGS:")
    for node_id, (color, building_type) in sorted(state.board.buildings.items()):
        btype = "S" if building_type == SETTLEMENT else "C"
        lines.append(f"  node {node_id}: {color.value} {btype}")

    # Roads
    seen_edges = set()
    road_lines = []
    for edge, color in state.board.roads.items():
        canonical = tuple(sorted(edge))
        if canonical not in seen_edges:
            seen_edges.add(canonical)
            road_lines.append(f"  edge {canonical}: {color.value}")
    if road_lines:
        lines.append("\nROADS:")
        lines.extend(road_lines)

    # Robber position
    lines.append(f"\nRobber at: {state.board.robber_coordinate}")

    return "\n".join(lines)


def serialize_actions(playable_actions) -> str:
    """Serialize available actions as an indexed list."""
    lines = ["AVAILABLE ACTIONS:"]
    for i, action in enumerate(playable_actions):
        action_str = _format_action(action)
        lines.append(f"  [{i}] {action_str}")
    return "\n".join(lines)


def _format_action(action) -> str:
    """Format a single action into a compact readable string."""
    at = action.action_type
    val = action.value

    if at == ActionType.ROLL:
        return "ROLL"
    elif at == ActionType.END_TURN:
        return "END_TURN"
    elif at == ActionType.BUILD_SETTLEMENT:
        return f"BUILD_SETTLEMENT node={val}"
    elif at == ActionType.BUILD_ROAD:
        return f"BUILD_ROAD edge={val}"
    elif at == ActionType.BUILD_CITY:
        return f"BUILD_CITY node={val}"
    elif at == ActionType.BUY_DEVELOPMENT_CARD:
        return "BUY_DEV_CARD"
    elif at == ActionType.PLAY_KNIGHT_CARD:
        return "PLAY_KNIGHT"
    elif at == ActionType.PLAY_YEAR_OF_PLENTY:
        resources = [RESOURCE_ABBREV.get(r, str(r)) for r in val]
        return f"YEAR_OF_PLENTY [{','.join(resources)}]"
    elif at == ActionType.PLAY_MONOPOLY:
        return f"MONOPOLY {RESOURCE_ABBREV.get(val, str(val))}"
    elif at == ActionType.PLAY_ROAD_BUILDING:
        return "ROAD_BUILDING"
    elif at == ActionType.MARITIME_TRADE:
        giving = [RESOURCE_ABBREV.get(r, "?") for r in val[:-1] if r is not None]
        receiving = RESOURCE_ABBREV.get(val[-1], str(val[-1]))
        return f"MARITIME_TRADE give=[{','.join(giving)}] get={receiving}"
    elif at == ActionType.MOVE_ROBBER:
        coord, target_color, _ = val
        target = target_color.value if target_color else "nobody"
        return f"MOVE_ROBBER to={coord} steal_from={target}"
    elif at == ActionType.DISCARD:
        if val:
            resources = [RESOURCE_ABBREV.get(r, str(r)) for r in val]
            return f"DISCARD [{','.join(resources)}]"
        return "DISCARD (random)"
    else:
        return f"{at.value} {val}"


def is_trade_eligible(playable_actions, game, player_color) -> bool:
    """Check if the current game state allows a domestic trade proposal.

    Eligible when: END_TURN is among actions (main play phase), not in
    special phases (discard, robber, road building), and player has resources.
    """
    has_end_turn = any(a.action_type == ActionType.END_TURN for a in playable_actions)
    if not has_end_turn:
        return False

    # Must not be roll phase
    if all(a.action_type == ActionType.ROLL for a in playable_actions):
        return False

    # Player must have at least 1 resource
    state = game.state
    key = player_key(state, player_color)
    total_resources = sum(state.player_state[f"{key}_{res}_IN_HAND"] for res in RESOURCES)
    return total_resources > 0


def _build_trade_option(game, player_color) -> str:
    """Build the DOMESTIC TRADE OPTION section for the user prompt."""
    state = game.state
    lines = [
        "DOMESTIC TRADE OPTION:",
        "Instead of choosing a numbered action, you may propose a domestic trade.",
        "Other players:",
    ]
    for color in state.colors:
        if color == player_color:
            continue
        key = player_key(state, color)
        total = sum(state.player_state[f"{key}_{res}_IN_HAND"] for res in RESOURCES)
        lines.append(f"  {color.value}: {total} cards")

    lines.append(
        'To propose: {"action": "trade", "target": "<COLOR>", '
        '"offering": {"W": 1}, "requesting": {"O": 1}, '
        '"reasoning": "...", "scratchpad": "..."}'
    )
    return "\n".join(lines)


def build_user_prompt(game, player_color, playable_actions, scratchpad: str = "", recent_events: list[str] | None = None, trade_eligible: bool = False) -> str:
    """Build the complete user prompt with dynamic state, events, scratchpad, and actions."""
    dynamic = serialize_dynamic_state(game, player_color)

    sections = [dynamic]

    # Recent events (last 8)
    if recent_events:
        events = recent_events[-8:]
        sections.append("RECENT EVENTS:\n" + "\n".join(f"  {e}" for e in events))

    # Scratchpad
    notes = scratchpad if scratchpad else "No notes yet."
    sections.append(f"YOUR NOTES (from previous turns):\n{notes}")

    actions = serialize_actions(playable_actions)
    sections.append(actions)

    # Domestic trade option (when eligible)
    if trade_eligible:
        sections.append(_build_trade_option(game, player_color))

    sections.append("Respond with ONLY a JSON object, no other text. Keep reasoning under 75 words. Example: {\"action\": 0, \"reasoning\": \"build toward port\", \"scratchpad\": \"need ore for cities\"}")

    return "\n\n".join(sections)
