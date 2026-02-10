"""LLM-backed Catanatron Player with auto-play, retry logic, and cost tracking."""

from __future__ import annotations

import logging

from catanatron.models.player import Player
from catanatron.models.enums import ActionType
from catanatron.models.actions import generate_playable_actions

from .state_serializer import serialize_system_prompt, build_user_prompt, is_trade_eligible
from .providers import LLMResponse, CostTracker
from .trading import (
    TradeProposal,
    TradeResult,
    build_responder_prompt,
    validate_resources,
    execute_trade,
)

logger = logging.getLogger(__name__)

# Actions that are never strategically interesting — auto-play if only option
AUTO_PLAY_ACTIONS = {ActionType.ROLL}

MAX_RETRIES = 3

# Sentinel action indices from providers (trade responses)
TRADE_PROPOSE = -1
TRADE_ACCEPT = -2
TRADE_REJECT = -3


class LLMPlayer(Player):
    """A Catanatron Player that delegates decisions to an LLM provider."""

    def __init__(self, color, provider, name: str | None = None, enable_trade: bool = True):
        super().__init__(color, is_bot=True)
        self.provider = provider
        self.name = name or provider.model_id
        self.enable_trade = enable_trade
        self.cost_tracker = CostTracker()
        self._system_prompt_cache: str | None = None
        self._calls_log: list[dict] = []
        self._scratchpad: str = ""
        self._recent_events: list[str] = []
        self._trade_used_this_turn: bool = False
        self._trade_results: list[tuple[int, TradeResult]] = []  # (turn, result)
        # Reference to all players, set by runner after game creation
        self._all_players_by_color: dict | None = None

    def decide(self, game, playable_actions):
        # Detect new turn (roll phase) and reset trade flag
        if playable_actions and playable_actions[0].action_type == ActionType.ROLL:
            self._trade_used_this_turn = False

        # Auto-play single-option turns (biggest cost optimization)
        if len(playable_actions) == 1:
            return playable_actions[0]

        # Auto-play rolls (never a strategic choice)
        if all(a.action_type == ActionType.ROLL for a in playable_actions):
            return playable_actions[0]

        # Build system prompt (cached across turns)
        if self._system_prompt_cache is None:
            self._system_prompt_cache = serialize_system_prompt(game, self.color)

        # Determine trade eligibility
        trade_eligible = (
            self.enable_trade
            and not self._trade_used_this_turn
            and is_trade_eligible(playable_actions, game, self.color)
        )

        user_prompt = build_user_prompt(
            game, self.color, playable_actions,
            scratchpad=self._scratchpad,
            recent_events=self._recent_events,
            trade_eligible=trade_eligible,
        )

        # Try up to MAX_RETRIES times
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                prompt = user_prompt
                if last_error and attempt > 0:
                    prompt += f"\n\n[RETRY: Your previous response was invalid: {last_error}. Please respond with valid JSON: {{\"action\": <index>, \"reasoning\": \"...\", \"scratchpad\": \"...\"}}]"

                response = self.provider.complete(
                    self._system_prompt_cache,
                    prompt,
                    max_tokens=1536,
                )
                self.cost_tracker.record(response)
                if response.scratchpad:
                    self._scratchpad = response.scratchpad[:2000]
                self._calls_log.append({
                    "turn": game.state.num_turns,
                    "action_index": response.action_index,
                    "action": None,  # filled in after validation
                    "reasoning": response.reasoning,
                    "scratchpad": response.scratchpad,
                    "latency_ms": response.latency_ms,
                    "cost_usd": response.cost_usd,
                    "cached_tokens": response.cached_tokens,
                })

                idx = response.action_index

                # Handle trade proposal
                if idx == TRADE_PROPOSE and trade_eligible and not self._trade_used_this_turn and response.trade_proposal:
                    self._calls_log[-1]["action"] = "TRADE_PROPOSAL"
                    result = self._handle_trade_proposal(
                        game, response.trade_proposal, response.reasoning, playable_actions
                    )
                    if result is not None:
                        # Trade was attempted — re-call decide() for the actual action
                        # (trade flag is already set, so no infinite recursion)
                        return self.decide(game, game.state.playable_actions)
                    # Trade failed validation — fall through to retry
                    last_error = "trade proposal was invalid, please choose a numbered action"
                    continue

                # Validate action index
                if 0 <= idx < len(playable_actions):
                    chosen = playable_actions[idx]
                    self._calls_log[-1]["action"] = f"{chosen.action_type.value} {chosen.value}"
                    logger.info(
                        f"{self.name} chose [{idx}] {chosen.action_type.value}: {response.reasoning}"
                    )
                    return chosen

                last_error = f"action index {idx} out of range [0, {len(playable_actions)-1}]"
                logger.warning(f"{self.name} attempt {attempt+1}: {last_error}")
                logger.warning(f"{self.name} raw response: {response.raw_text[:300]}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"{self.name} attempt {attempt+1} error: {e}")

        # Fallback: pick first non-END_TURN action, or first action
        logger.error(f"{self.name} failed all retries, using fallback")
        for action in playable_actions:
            if action.action_type != ActionType.END_TURN:
                return action
        return playable_actions[0]

    def _handle_trade_proposal(self, game, trade_proposal: dict, reasoning: str, playable_actions) -> TradeResult | None:
        """Process a trade proposal from the LLM response.

        Returns TradeResult if trade was attempted, None if validation failed.
        """
        self._trade_used_this_turn = True

        target_color_str = trade_proposal.get("target", "")
        offering = trade_proposal.get("offering", {})
        requesting = trade_proposal.get("requesting", {})

        # Validate proposer has offered resources
        if not validate_resources(game.state, self.color, offering):
            logger.warning(f"{self.name}: trade proposal invalid — insufficient resources to offer")
            return None

        # Find target player
        target_player = None
        if self._all_players_by_color:
            for color, player in self._all_players_by_color.items():
                if color.value.upper() == target_color_str.upper():
                    target_player = player
                    break

        if target_player is None:
            logger.warning(f"{self.name}: invalid trade target '{target_color_str}'")
            return None

        # Check target is an LLMPlayer
        if not isinstance(target_player, LLMPlayer):
            logger.warning(f"{self.name}: trade target {target_color_str} is not an LLM player")
            return None

        proposal = TradeProposal(
            proposer_color=self.color.value,
            target_color=target_player.color.value,
            offering=offering,
            requesting=requesting,
        )

        # Call responder
        accepted, responder_reasoning = self._call_responder(
            game, target_player, offering, requesting
        )

        if accepted:
            # Validate responder has requested resources
            if not validate_resources(game.state, target_player.color, requesting):
                logger.warning(f"{self.name}: responder {target_color_str} lacks requested resources")
                result = TradeResult(
                    proposal=proposal,
                    accepted=False,
                    proposer_reasoning=reasoning,
                    responder_reasoning="Invalid: insufficient resources",
                )
                self._trade_results.append((game.state.num_turns, result))
                return result

            # Execute trade
            execute_trade(game.state, self.color, target_player.color, offering, requesting)
            # Regenerate playable actions after trade
            game.state.playable_actions = generate_playable_actions(game.state)

            offering_str = ", ".join(f"{v} {k}" for k, v in offering.items() if v and int(v) > 0)
            requesting_str = ", ".join(f"{v} {k}" for k, v in requesting.items() if v and int(v) > 0)
            logger.info(
                f"Trade accepted: {self.color.value} gives {offering_str} to "
                f"{target_player.color.value} for {requesting_str}"
            )

        result = TradeResult(
            proposal=proposal,
            accepted=accepted,
            proposer_reasoning=reasoning,
            responder_reasoning=responder_reasoning,
        )
        self._trade_results.append((game.state.num_turns, result))
        return result

    def _call_responder(self, game, target_player: LLMPlayer, offering: dict, requesting: dict) -> tuple[bool, str]:
        """Call the responder LLM to accept/reject the trade.

        Returns (accepted: bool, reasoning: str).
        """
        # Ensure target has a cached system prompt
        if target_player._system_prompt_cache is None:
            target_player._system_prompt_cache = serialize_system_prompt(game, target_player.color)

        user_prompt = build_responder_prompt(
            game,
            responder_color=target_player.color,
            proposer_color_str=self.color.value,
            offering=offering,
            requesting=requesting,
            scratchpad=target_player._scratchpad,
            recent_events=target_player._recent_events,
        )

        try:
            response = target_player.provider.complete(
                target_player._system_prompt_cache,
                user_prompt,
                max_tokens=256,
            )
            target_player.cost_tracker.record(response)
            target_player._calls_log.append({
                "turn": game.state.num_turns,
                "action_index": response.action_index,
                "action": "TRADE_RESPONSE",
                "reasoning": response.reasoning,
                "scratchpad": response.scratchpad,
                "latency_ms": response.latency_ms,
                "cost_usd": response.cost_usd,
                "cached_tokens": response.cached_tokens,
            })
            if response.scratchpad:
                target_player._scratchpad = response.scratchpad[:2000]

            accepted = response.action_index == TRADE_ACCEPT
            return accepted, response.reasoning

        except Exception as e:
            logger.warning(f"Trade response error from {target_player.name}: {e}")
            return False, f"Error: {e}"

    def reset_state(self):
        """Reset between games, keep cost tracker."""
        self._system_prompt_cache = None
        self._calls_log = []
        self._scratchpad = ""
        self._recent_events = []
        self._trade_used_this_turn = False
        self._trade_results = []
        self._all_players_by_color = None

    def __repr__(self):
        return f"LLMPlayer({self.name}:{self.color.value})"
