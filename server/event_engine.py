"""
AI Policy Engine — Seeded Random Event Engine

Generates reproducible, dramatic world events (pandemics, booms,
climate crises, protests, …) that stress-test the agent's ability
to react and recover.

Events are controlled by a seeded RNG for full reproducibility.
Each event has a per-step trigger probability, a duration, and
per-step deltas applied while the event is active.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# Event Definitions
# ─────────────────────────────────────────────────────────────

@dataclass
class EventType:
    """Template for a category of random event."""

    name: str
    description: str
    base_probability: float          # per-step chance (before multiplier)
    duration: int                    # how many steps it lasts
    per_step_deltas: Dict[str, float]  # applied EACH step while active
    onset_deltas: Dict[str, float] = field(default_factory=dict)  # one-time on trigger


EVENT_TYPES: List[EventType] = [
    EventType(
        name="pandemic",
        description="A global pandemic strains healthcare and the economy.",
        base_probability=0.025,
        duration=5,
        per_step_deltas={
            "gdp_index": -2.5,
            "public_satisfaction": -3.0,
            "unemployment_rate": 1.5,
            "healthcare_index": -2.0,
        },
        onset_deltas={
            "public_satisfaction": -8.0,
            "foreign_investment": -5.0,
        },
    ),
    EventType(
        name="industrial_boom",
        description="An industrial boom drives rapid growth — and pollution.",
        base_probability=0.045,
        duration=3,
        per_step_deltas={
            "gdp_index": 2.5,
            "industrial_output": 3.0,
            "pollution_index": 4.0,
            "unemployment_rate": -1.5,
            "carbon_emission_rate": 2.0,
        },
    ),
    EventType(
        name="climate_crisis",
        description="Extreme weather destabilises the ecosystem and public mood.",
        base_probability=0.035,
        duration=4,
        per_step_deltas={
            "pollution_index": 6.0,
            "ecological_stability": -5.0,
            "public_satisfaction": -2.0,
        },
        onset_deltas={
            "pollution_index": 10.0,
            "transport_efficiency": -5.0,
        },
    ),
    EventType(
        name="public_protest",
        description="Widespread protests demand policy change.",
        base_probability=0.055,
        duration=2,
        per_step_deltas={
            "public_satisfaction": -4.0,
            "foreign_investment": -2.5,
            "regulation_strength": 1.5,
        },
        onset_deltas={
            "public_satisfaction": -5.0,
        },
    ),
    EventType(
        name="tech_breakthrough",
        description="A clean-energy breakthrough accelerates green transition.",
        base_probability=0.035,
        duration=3,
        per_step_deltas={
            "renewable_energy_ratio": 0.025,
            "energy_efficiency": 2.5,
            "gdp_index": 1.0,
        },
        onset_deltas={
            "public_satisfaction": 5.0,
        },
    ),
    EventType(
        name="trade_war",
        description="International trade tensions disrupt commerce.",
        base_probability=0.030,
        duration=4,
        per_step_deltas={
            "trade_balance": -3.5,
            "foreign_investment": -2.5,
            "gdp_index": -1.5,
            "industrial_output": -1.0,
        },
        onset_deltas={
            "trade_balance": -8.0,
        },
    ),
    EventType(
        name="natural_disaster",
        description="A major natural disaster damages infrastructure.",
        base_probability=0.025,
        duration=2,
        per_step_deltas={
            "gdp_index": -3.0,
            "transport_efficiency": -4.0,
            "public_satisfaction": -5.0,
            "pollution_index": 4.0,
        },
        onset_deltas={
            "gdp_index": -5.0,
            "transport_efficiency": -5.0,
        },
    ),
    EventType(
        name="economic_recession",
        description="A recession contracts the economy and labour market.",
        base_probability=0.025,
        duration=5,
        per_step_deltas={
            "gdp_index": -2.0,
            "unemployment_rate": 1.2,
            "industrial_output": -1.5,
            "foreign_investment": -1.5,
            "inflation_rate": -0.5,
        },
        onset_deltas={
            "public_satisfaction": -6.0,
        },
    ),
]


# ─────────────────────────────────────────────────────────────
# Active Event Instance
# ─────────────────────────────────────────────────────────────

@dataclass
class ActiveEvent:
    """A currently running event instance."""

    event_type: EventType
    remaining_steps: int


# ─────────────────────────────────────────────────────────────
# Event Engine
# ─────────────────────────────────────────────────────────────

class EventEngine:
    """
    Manages random event triggering, tracking, and application.

    Args:
        seed: Random seed for reproducibility.
        frequency_multiplier: Scales all event probabilities
            (0.0 = events disabled, 1.0 = normal, 2.0 = chaotic).
    """

    def __init__(self, seed: int = 42, frequency_multiplier: float = 1.0,
                 satisfaction_event_scale: float = 1.0):
        self._rng = random.Random(seed)
        self._freq_mult = frequency_multiplier
        self._sat_scale = satisfaction_event_scale
        self._active_events: List[ActiveEvent] = []

    def reset(self, seed: int = 42, frequency_multiplier: float = 1.0,
              satisfaction_event_scale: float = 1.0) -> None:
        """Reinitialise the engine for a new episode."""
        self._rng = random.Random(seed)
        self._freq_mult = frequency_multiplier
        self._sat_scale = satisfaction_event_scale
        self._active_events.clear()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def step(self, state: Dict[str, float]) -> List[str]:
        """
        Advance one step: trigger new events, apply active events, expire old.

        Returns:
            List of names of currently active events (after this step).
        """
        self._trigger_new_events(state)
        self._apply_active_events(state)
        self._expire_events()
        return self.active_event_names

    @property
    def active_event_names(self) -> List[str]:
        return [ae.event_type.name for ae in self._active_events]

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _trigger_new_events(self, state: Dict[str, float]) -> None:
        """Roll dice for each event type; start new ones on success."""
        if self._freq_mult <= 0:
            return

        # Active event names to prevent duplicates
        active_names = {ae.event_type.name for ae in self._active_events}

        for et in EVENT_TYPES:
            if et.name in active_names:
                continue  # don't stack identical events

            prob = et.base_probability * self._freq_mult

            # Context-sensitive probability modifiers
            prob = self._adjust_probability(et, state, prob)

            if self._rng.random() < prob:
                self._active_events.append(
                    ActiveEvent(event_type=et, remaining_steps=et.duration)
                )
                # Apply one-time onset deltas
                for key, delta in et.onset_deltas.items():
                    d = delta * self._sat_scale if key == "public_satisfaction" else delta
                    state[key] = state.get(key, 0.0) + d

    def _adjust_probability(
        self, et: EventType, state: Dict[str, float], base_prob: float
    ) -> float:
        """Make certain events more likely when conditions are ripe."""
        prob = base_prob

        if et.name == "public_protest" and state.get("public_satisfaction", 50) < 30:
            prob *= 2.0  # unrest breeds protest

        if et.name == "climate_crisis" and state.get("pollution_index", 100) > 180:
            prob *= 1.8  # high pollution → more climate events

        if et.name == "economic_recession" and state.get("inflation_rate", 3) > 12:
            prob *= 1.5  # high inflation → recession risk

        if et.name == "tech_breakthrough" and state.get("green_subsidies", 10) > 40:
            prob *= 1.6  # more R&D funding → more breakthroughs

        if et.name == "pandemic" and state.get("healthcare_index", 50) < 25:
            prob *= 1.8  # weak healthcare → worse pandemic risk

        return min(prob, 0.30)  # cap at 30 % per step

    def _apply_active_events(self, state: Dict[str, float]) -> None:
        """Apply per-step deltas of every active event."""
        for ae in self._active_events:
            for key, delta in ae.event_type.per_step_deltas.items():
                d = delta * self._sat_scale if key == "public_satisfaction" else delta
                state[key] = state.get(key, 0.0) + d

    def _expire_events(self) -> None:
        """Decrement remaining steps; remove expired events."""
        for ae in self._active_events:
            ae.remaining_steps -= 1
        self._active_events = [ae for ae in self._active_events if ae.remaining_steps > 0]
