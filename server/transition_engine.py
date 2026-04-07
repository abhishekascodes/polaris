"""
AI Policy Engine — Multi-Layer Transition Engine

Implements 4 distinct transition layers that execute sequentially
each step, producing emergent, realistic dynamics:

  Layer 1 — Deterministic: Direct, immediate effects of each action.
  Layer 2 — Non-linear:    Threshold-based exponential / quadratic effects.
  Layer 3 — Delayed:       Queued effects that materialise after N steps.
  Layer 4 — Feedback:      Systemic loops (pollution→health→productivity→GDP).

After all layers, metric values are clamped to their configured bounds.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

from .config import STATE_BOUNDS


# =====================================================================
# Types
# =====================================================================

# A delayed effect: (step_when_it_fires, dict_of_deltas)
DelayedEffect = Tuple[int, Dict[str, float]]


class TransitionEngine:
    """
    Applies all four transition layers to advance the world state by one step.

    Usage:
        engine = TransitionEngine()
        engine.apply(state, action, current_step)
    """

    def __init__(self) -> None:
        self._delayed_queue: Deque[DelayedEffect] = deque()

    def reset(self) -> None:
        """Clear all delayed effects (called on environment reset)."""
        self._delayed_queue.clear()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def apply(self, state: Dict[str, float], action: str, step: int) -> None:
        """Apply all four layers in order, then clamp to bounds."""
        self._layer1_deterministic(state, action, step)
        self._layer2_nonlinear(state)
        self._layer3_delayed(state, step)
        self._layer4_feedback(state)
        self._clamp(state)

    # =================================================================
    # Layer 1 — Deterministic Effects
    # =================================================================

    def _layer1_deterministic(
        self, s: Dict[str, float], action: str, step: int
    ) -> None:
        if action == "no_action":
            return

        if action == "increase_tax":
            s["tax_rate"] += 3.0
            s["gdp_index"] -= 1.5
            s["industrial_output"] -= 2.0
            s["foreign_investment"] -= 3.0
            s["welfare_spending"] += 2.0
            s["public_satisfaction"] -= 2.0

        elif action == "decrease_tax":
            s["tax_rate"] -= 3.0
            s["gdp_index"] += 1.5
            s["industrial_output"] += 2.0
            s["foreign_investment"] += 3.0
            s["welfare_spending"] -= 2.0
            s["inequality_index"] += 1.5

        elif action == "stimulate_economy":
            s["gdp_index"] += 3.0
            s["industrial_output"] += 2.0
            s["inflation_rate"] += 1.5
            s["unemployment_rate"] -= 2.0
            s["pollution_index"] += 5.0
            s["carbon_emission_rate"] += 3.0

        elif action == "reduce_interest_rates":
            s["interest_rate"] -= 1.0
            s["gdp_index"] += 2.0
            s["inflation_rate"] += 1.0
            s["foreign_investment"] -= 2.0
            s["trade_balance"] -= 2.0

        elif action == "expand_industry":
            s["industrial_output"] += 5.0
            s["gdp_index"] += 3.0
            s["pollution_index"] += 8.0
            s["carbon_emission_rate"] += 5.0
            s["unemployment_rate"] -= 3.0

        elif action == "restrict_polluting_industries":
            s["industrial_output"] -= 4.0
            s["pollution_index"] -= 6.0
            s["carbon_emission_rate"] -= 5.0
            s["unemployment_rate"] += 1.5
            s["gdp_index"] -= 2.0
            s["public_satisfaction"] += 0.5  # public approves green action

        elif action == "incentivize_clean_tech":
            s["gdp_index"] -= 1.0
            s["green_subsidies"] += 5.0
            # Delayed green tech payoff
            self._enqueue_delayed(step + 3, {
                "renewable_energy_ratio": 0.03,
                "pollution_index": -3.0,
                "energy_efficiency": 2.0,
            })

        elif action == "enforce_emission_limits":
            s["regulation_strength"] += 5.0
            s["pollution_index"] -= 4.0
            s["carbon_emission_rate"] -= 4.0
            s["industrial_output"] -= 2.0
            s["foreign_investment"] -= 2.0
            s["public_satisfaction"] += 0.5  # public approves clean air

        elif action == "subsidize_renewables":
            s["green_subsidies"] += 8.0
            s["gdp_index"] -= 1.5
            # Larger delayed green payoff
            self._enqueue_delayed(step + 4, {
                "renewable_energy_ratio": 0.05,
                "pollution_index": -5.0,
                "energy_efficiency": 3.0,
                "carbon_emission_rate": -3.0,
            })

        elif action == "implement_carbon_tax":
            s["pollution_index"] -= 5.0
            s["carbon_emission_rate"] -= 6.0
            s["industrial_output"] -= 3.0
            s["gdp_index"] -= 2.0
            s["foreign_investment"] -= 3.0
            s["tax_rate"] += 2.0
            s["public_satisfaction"] += 0.3  # moderate green approval

        elif action == "increase_welfare":
            s["welfare_spending"] += 5.0
            s["public_satisfaction"] += 3.0
            s["inequality_index"] -= 2.0
            s["gdp_index"] -= 1.0

        elif action == "invest_in_healthcare":
            s["gdp_index"] -= 1.0
            s["public_satisfaction"] += 1.0
            self._enqueue_delayed(step + 2, {
                "healthcare_index": 5.0,
                "public_satisfaction": 1.5,
            })

        elif action == "invest_in_education":
            s["gdp_index"] -= 1.0
            self._enqueue_delayed(step + 3, {
                "education_index": 4.0,
                "inequality_index": -1.0,
            })
            # Long-term innovation dividend
            self._enqueue_delayed(step + 6, {
                "gdp_index": 2.0,
                "industrial_output": 1.0,
            })

        elif action == "upgrade_energy_grid":
            s["gdp_index"] -= 1.5
            self._enqueue_delayed(step + 3, {
                "energy_efficiency": 6.0,
                "renewable_energy_ratio": 0.03,
                "pollution_index": -2.0,
            })

        elif action == "invest_in_transport":
            s["gdp_index"] -= 1.0
            self._enqueue_delayed(step + 3, {
                "transport_efficiency": 5.0,
                "gdp_index": 1.0,
                "pollution_index": -2.0,
            })

    # =================================================================
    # Layer 2 — Non-linear / Threshold Effects
    # =================================================================

    def _layer2_nonlinear(self, s: Dict[str, float]) -> None:
        # ── Pollution catastrophe ──
        if s["pollution_index"] > 200:
            excess = s["pollution_index"] - 200
            s["healthcare_index"] -= excess * 0.15
            s["ecological_stability"] -= excess * 0.10
        if s["pollution_index"] > 250:
            excess = s["pollution_index"] - 250
            s["public_satisfaction"] -= excess * 0.30

        # ── Tax over-burden ──
        if s["tax_rate"] > 40:
            excess = s["tax_rate"] - 40
            s["gdp_index"] -= (excess ** 1.5) * 0.10
            s["industrial_output"] -= excess * 0.50
            s["foreign_investment"] -= excess * 0.80

        # ── Mass unemployment crisis ──
        if s["unemployment_rate"] > 25:
            excess = s["unemployment_rate"] - 25
            s["public_satisfaction"] -= excess * 1.50
            s["inequality_index"] += excess * 0.30

        # ── Hyper-inflation ──
        if s["inflation_rate"] > 15:
            excess = s["inflation_rate"] - 15
            s["public_satisfaction"] -= excess * 1.0
            s["foreign_investment"] -= excess * 0.50

        # ── GDP depression spiral ──
        if s["gdp_index"] < 40:
            deficit = 40 - s["gdp_index"]
            s["unemployment_rate"] += deficit * 0.20
            s["public_satisfaction"] -= deficit * 0.15

        # ── Ecological tipping point ──
        if s["ecological_stability"] < 20:
            deficit = 20 - s["ecological_stability"]
            s["pollution_index"] += deficit * 0.25
            s["public_satisfaction"] -= deficit * 0.20

    # =================================================================
    # Layer 3 — Delayed Effects
    # =================================================================

    def _layer3_delayed(self, s: Dict[str, float], step: int) -> None:
        """Fire any delayed effects whose step has arrived."""
        remaining: Deque[DelayedEffect] = deque()
        while self._delayed_queue:
            fire_step, deltas = self._delayed_queue.popleft()
            if step >= fire_step:
                for key, delta in deltas.items():
                    s[key] = s.get(key, 0.0) + delta
            else:
                remaining.append((fire_step, deltas))
        self._delayed_queue = remaining

    def _enqueue_delayed(self, fire_step: int, deltas: Dict[str, float]) -> None:
        self._delayed_queue.append((fire_step, deltas))

    # =================================================================
    # Layer 4 — Feedback Loops (emergent dynamics)
    # =================================================================

    def _layer4_feedback(self, s: Dict[str, float]) -> None:
        # ── Health–Productivity loop ──
        if s["healthcare_index"] < 30:
            deficit = 30 - s["healthcare_index"]
            s["industrial_output"] -= deficit * 0.08
            s["unemployment_rate"] += 0.30

        # ── Education–Innovation loop ──
        if s["education_index"] > 70:
            s["gdp_index"] += 0.35
            s["inequality_index"] -= 0.10

        # ── Inequality–Satisfaction loop ──
        if s["inequality_index"] > 60:
            excess = s["inequality_index"] - 60
            s["public_satisfaction"] -= excess * 0.10

        # ── Unemployment–Social loop (softened) ──
        if s["unemployment_rate"] > 18:
            excess = s["unemployment_rate"] - 18
            s["public_satisfaction"] -= excess * 0.18
            s["inequality_index"] += 0.15

        # ── Pollution–Health loop ──
        if s["pollution_index"] > 150:
            excess = s["pollution_index"] - 150
            s["healthcare_index"] -= excess * 0.04

        # ── Renewable energy dividend ──
        rer = s.get("renewable_energy_ratio", 0.0)
        if rer > 0.3:
            green_bonus = (rer - 0.3) * 8.0
            s["pollution_index"] -= green_bonus * 0.40
            s["energy_efficiency"] += green_bonus * 0.10
            s["carbon_emission_rate"] -= green_bonus * 0.30

        # ── Satisfaction–Stability pressure ──
        if s["public_satisfaction"] < 20:
            s["regulation_strength"] += 0.8
            s["foreign_investment"] -= 0.8

        # ── Natural carbon cycle (slow regeneration) ──
        if s["pollution_index"] > 0:
            s["pollution_index"] -= 0.5  # nature absorbs some pollution each turn
        if s["ecological_stability"] < 100 and s["pollution_index"] < 80:
            s["ecological_stability"] += 0.3  # slow ecosystem recovery

        # ── Natural satisfaction drift (regression toward 45) ──
        # Prevents trivial spirals from modest unemployment
        sat = s["public_satisfaction"]
        baseline = 45.0
        if sat < baseline:
            s["public_satisfaction"] += min(0.6, (baseline - sat) * 0.03)
        elif sat > 80:
            s["public_satisfaction"] -= (sat - 80) * 0.01  # diminishing returns

    # =================================================================
    # Clamping
    # =================================================================

    @staticmethod
    def _clamp(s: Dict[str, float]) -> None:
        for key, (lo, hi) in STATE_BOUNDS.items():
            if key in s:
                s[key] = max(lo, min(hi, s[key]))
