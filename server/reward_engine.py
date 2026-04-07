"""
AI Policy Engine — Multi-Objective Reward Engine

Computes a dense, per-step reward signal composed of four sub-scores
(economic, environmental, social, stability) plus penalties for
destructive behaviour and policy oscillation.

The reward provides useful gradient signal across the entire trajectory
(not just sparse end-of-episode), enabling effective RL training.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .config import KEY_METRICS, REWARD_WEIGHTS, STATE_BOUNDS


def _normalise(value: float, lo: float, hi: float) -> float:
    """Normalise *value* to [0, 1] given known bounds."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _inv_normalise(value: float, lo: float, hi: float) -> float:
    """Inverse normalise: 1.0 when value == lo, 0.0 when value == hi."""
    return 1.0 - _normalise(value, lo, hi)


class RewardEngine:
    """
    Calculates multi-objective reward each step.

    Tracks a short history window for volatility / oscillation detection.
    """

    def __init__(self) -> None:
        self._prev_actions: List[str] = []

    def reset(self) -> None:
        self._prev_actions.clear()

    def compute(
        self,
        state: Dict[str, float],
        prev_state: Optional[Dict[str, float]],
        action: str,
    ) -> Dict[str, float]:
        """
        Return a breakdown dict with individual sub-scores and the total reward.

        Keys: economic_score, environmental_score, social_score,
              stability_score, penalties, total_reward.
        """
        econ = self._economic_score(state)
        envr = self._environmental_score(state)
        soc = self._social_score(state)
        stab = self._stability_score(state, prev_state)

        weighted = (
            REWARD_WEIGHTS["economic"] * econ
            + REWARD_WEIGHTS["environmental"] * envr
            + REWARD_WEIGHTS["social"] * soc
            + REWARD_WEIGHTS["stability"] * stab
        )

        penalties = self._compute_penalties(state, action)

        total = round(max(-1.0, min(1.0, weighted - penalties)), 4)

        # Track action history for oscillation detection
        self._prev_actions.append(action)
        if len(self._prev_actions) > 6:
            self._prev_actions.pop(0)

        return {
            "economic_score": round(econ, 4),
            "environmental_score": round(envr, 4),
            "social_score": round(soc, 4),
            "stability_score": round(stab, 4),
            "penalties": round(penalties, 4),
            "total_reward": total,
        }

    # -----------------------------------------------------------------
    # Sub-scores (each 0–1)
    # -----------------------------------------------------------------

    @staticmethod
    def _economic_score(s: Dict[str, float]) -> float:
        gdp = _normalise(s["gdp_index"], *STATE_BOUNDS["gdp_index"])
        unemp = _inv_normalise(s["unemployment_rate"], *STATE_BOUNDS["unemployment_rate"])
        ind = _normalise(s["industrial_output"], *STATE_BOUNDS["industrial_output"])
        inv = _normalise(s["foreign_investment"], *STATE_BOUNDS["foreign_investment"])
        trade = _normalise(
            s["trade_balance"] + 100, 0, 200  # shift to 0–200 range
        )
        return 0.35 * gdp + 0.25 * unemp + 0.20 * ind + 0.10 * inv + 0.10 * trade

    @staticmethod
    def _environmental_score(s: Dict[str, float]) -> float:
        poll = _inv_normalise(s["pollution_index"], *STATE_BOUNDS["pollution_index"])
        rer = _normalise(s["renewable_energy_ratio"], *STATE_BOUNDS["renewable_energy_ratio"])
        eco = _normalise(s["ecological_stability"], *STATE_BOUNDS["ecological_stability"])
        carb = _inv_normalise(s["carbon_emission_rate"], *STATE_BOUNDS["carbon_emission_rate"])
        return 0.35 * poll + 0.25 * rer + 0.25 * eco + 0.15 * carb

    @staticmethod
    def _social_score(s: Dict[str, float]) -> float:
        sat = _normalise(s["public_satisfaction"], *STATE_BOUNDS["public_satisfaction"])
        hc = _normalise(s["healthcare_index"], *STATE_BOUNDS["healthcare_index"])
        edu = _normalise(s["education_index"], *STATE_BOUNDS["education_index"])
        ineq = _inv_normalise(s["inequality_index"], *STATE_BOUNDS["inequality_index"])
        return 0.35 * sat + 0.25 * hc + 0.20 * edu + 0.20 * ineq

    @staticmethod
    def _stability_score(
        s: Dict[str, float],
        prev: Optional[Dict[str, float]],
    ) -> float:
        """Low volatility → high stability score."""
        if prev is None:
            return 0.8  # generous default on first step

        total_change = 0.0
        for key in KEY_METRICS:
            lo, hi = STATE_BOUNDS.get(key, (0, 100))
            span = hi - lo if hi > lo else 1.0
            delta = abs(s.get(key, 0) - prev.get(key, 0)) / span
            total_change += delta

        avg_change = total_change / len(KEY_METRICS)
        # 0 change → 1.0,  avg_change ≥ 0.15 → 0.0
        return max(0.0, min(1.0, 1.0 - avg_change / 0.15))

    # -----------------------------------------------------------------
    # Penalties
    # -----------------------------------------------------------------

    def _compute_penalties(self, state: Dict[str, float], action: str) -> float:
        p = 0.0

        # ── Policy oscillation (ABAB pattern) ──
        if len(self._prev_actions) >= 4:
            last4 = self._prev_actions[-4:]
            if last4[0] == last4[2] and last4[1] == last4[3] and last4[0] != last4[1]:
                p += 0.05

        # ── Rapid flip-flop (undo previous action) ──
        opposite_pairs = {
            "increase_tax": "decrease_tax",
            "decrease_tax": "increase_tax",
            "expand_industry": "restrict_polluting_industries",
            "restrict_polluting_industries": "expand_industry",
        }
        if len(self._prev_actions) >= 1:
            prev = self._prev_actions[-1]
            if opposite_pairs.get(action) == prev:
                p += 0.03

        # ── Collapse proximity ──
        if state["gdp_index"] < 30:
            p += (30 - state["gdp_index"]) * 0.005
        if state["pollution_index"] > 260:
            p += (state["pollution_index"] - 260) * 0.003
        if state["public_satisfaction"] < 15:
            p += (15 - state["public_satisfaction"]) * 0.005

        # ── Extreme inaction under crisis ──
        crisis = (
            state["pollution_index"] > 220
            or state["gdp_index"] < 35
            or state["public_satisfaction"] < 15
        )
        if crisis and action == "no_action":
            p += 0.04

        return p
