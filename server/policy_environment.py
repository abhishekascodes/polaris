"""
AI Policy Engine — Core Environment Implementation

Implements the OpenEnv Environment interface:
  reset(seed, episode_id, **kwargs) → Observation
  step(action)                      → Observation
  state (property)                  → State

The environment orchestrates the transition engine, event engine,
and reward engine to produce a cohesive governance simulation.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Standalone / inference-only fallback
    from pydantic import BaseModel, Field

    class Action(BaseModel):  # type: ignore[no-redef]
        metadata: dict = Field(default_factory=dict)

    class Observation(BaseModel):  # type: ignore[no-redef]
        done: bool = False
        reward: float = 0.0
        metadata: dict = Field(default_factory=dict)

    class State(BaseModel):  # type: ignore[no-redef]
        episode_id: Optional[str] = None
        step_count: int = 0

    class Environment:  # type: ignore[no-redef]
        """Minimal base when openenv-core is unavailable."""
        def reset(self, **kw: Any) -> Observation:
            raise NotImplementedError
        def step(self, action: Action, **kw: Any) -> Observation:
            raise NotImplementedError
        @property
        def state(self) -> State:
            raise NotImplementedError
        def close(self) -> None:
            pass
        def reset_async(self, **kw: Any) -> Observation:
            return self.reset(**kw)
        def step_async(self, action: Action, **kw: Any) -> Observation:
            return self.step(action, **kw)

from .config import (
    COLLAPSE_CONDITIONS,
    DEFAULT_STATE,
    TASK_CONFIGS,
    VALID_ACTIONS,
    ACTION_DESCRIPTIONS,
)
from .event_engine import EventEngine
from .explainability import ExplainabilityEngine
from .reward_engine import RewardEngine
from .tasks import grade_trajectory
from .transition_engine import TransitionEngine


class PolicyEnvironment(Environment):
    """
    AI Policy Engine — multi-objective governance simulation.

    The agent acts as a national policy-maker, choosing one of 16
    policy levers each step.  The world state evolves through four
    transition layers (deterministic, non-linear, delayed, feedback)
    plus a seeded random event system.  A multi-objective reward
    signal balances economic, environmental, social and stability goals.

    Three tasks of increasing difficulty test progressively harder
    aspects of the environment.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._state_obj = State(episode_id=str(uuid4()), step_count=0)
        self._world: Dict[str, float] = {}
        self._prev_world: Optional[Dict[str, float]] = None
        self._last_actions: List[str] = []
        self._trajectory: List[Dict] = []

        self._task_id: str = "environmental_recovery"
        self._max_steps: int = 50
        self._done: bool = False

        self._transition = TransitionEngine()
        self._events = EventEngine()
        self._reward_eng = RewardEngine()
        self._explainer = ExplainabilityEngine()

    # =================================================================
    # reset()
    # =================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Initialise (or re-initialise) the environment for a new episode.

        Args:
            seed: Random seed for reproducible event generation.
            episode_id: Custom episode identifier.
            task_id: Which task to run. Defaults to "environmental_recovery".
        """
        # Resolve task
        self._task_id = task_id or kwargs.get("task_id", "environmental_recovery")
        if self._task_id not in TASK_CONFIGS:
            self._task_id = "environmental_recovery"

        cfg = TASK_CONFIGS[self._task_id]
        self._task_cfg = cfg
        self._max_steps = cfg["max_steps"]

        # Build initial world state
        self._world = copy.deepcopy(DEFAULT_STATE)
        for key, val in cfg.get("initial_state_overrides", {}).items():
            self._world[key] = val

        self._prev_world = None
        self._last_actions = []
        self._trajectory = []
        self._done = False

        # Reset sub-engines
        real_seed = seed if seed is not None else 42
        self._transition.reset()
        self._events.reset(
            seed=real_seed,
            frequency_multiplier=cfg.get("event_frequency_multiplier", 1.0),
            satisfaction_event_scale=cfg.get("satisfaction_event_scale", 1.0),
        )
        self._reward_eng.reset()

        # Episode metadata
        self._state_obj = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        obs_meta = self._build_observation_metadata(reward_breakdown=None)
        self._trajectory.append(copy.deepcopy(obs_meta))

        return Observation(done=False, reward=0.0, metadata=obs_meta)

    # =================================================================
    # step()
    # =================================================================

    def step(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a single policy action and advance the simulation by one step.

        Args:
            action: A PolicyAction (or Action with action field / metadata).
        """
        if self._done:
            obs_meta = self._build_observation_metadata(reward_breakdown=None)
            return Observation(done=True, reward=0.0, metadata=obs_meta)

        # --- Parse action string ---
        action_str = self._parse_action(action)

        # --- Snapshot previous state ---
        self._prev_world = copy.deepcopy(self._world)

        # --- Apply transition engine (Layers 1-4) ---
        self._transition.apply(
            self._world, action_str, self._state_obj.step_count
        )

        # --- Satisfaction floor damping (calibrated regime) ---
        # Creates an asymptotic "sticky floor" near the collapse threshold.
        # Smart agents that pump welfare in this zone can pull back;
        # random agents still collapse because they don't prioritise welfare.
        floor_damp = self._task_cfg.get("satisfaction_floor_damping", 0)

        if floor_damp > 0:
            # Phase 1: dampen feedback loop losses (from transition engine)
            sat_post_trans = self._world.get("public_satisfaction", 50)
            sat_prev = self._prev_world.get("public_satisfaction", 50)
            if sat_post_trans < 35 and sat_post_trans < sat_prev:
                loss = sat_prev - sat_post_trans
                gradient = min(1.0, floor_damp + (35 - sat_post_trans) / 35 * 0.15)
                self._world["public_satisfaction"] = sat_prev - loss * (1.0 - gradient)

        # --- Apply event engine ---
        sat_pre_events = self._world.get("public_satisfaction", 50)
        active_events = self._events.step(self._world)

        if floor_damp > 0:
            # Phase 2: dampen event shock losses
            sat_post_events = self._world.get("public_satisfaction", 50)
            if sat_post_events < 35 and sat_post_events < sat_pre_events:
                loss = sat_pre_events - sat_post_events
                gradient = min(1.0, floor_damp + (35 - sat_post_events) / 35 * 0.15)
                self._world["public_satisfaction"] = sat_pre_events - loss * (1.0 - gradient)

        # --- Emergency crisis response bonus ---
        # Social investment during low satisfaction gets extra effectiveness.
        # This rewards crisis-responsive policy-making: smart agents that
        # prioritise social stability during crises over other objectives.
        crisis_bonus = self._task_cfg.get("crisis_welfare_bonus", 0)
        social_actions = {"increase_welfare", "invest_in_healthcare", "invest_in_education"}
        if crisis_bonus > 0 and action_str in social_actions:
            sat_cur = self._world.get("public_satisfaction", 50)
            if sat_cur < 40:
                # Welfare gets full bonus, healthcare/education get partial
                bonus = crisis_bonus if action_str == "increase_welfare" else crisis_bonus * 0.5
                self._world["public_satisfaction"] += bonus

        # --- Re-clamp after events ---
        TransitionEngine._clamp(self._world)

        # --- Track action history ---
        self._last_actions.append(action_str)
        if len(self._last_actions) > 5:
            self._last_actions.pop(0)

        # --- Compute reward ---
        reward_info = self._reward_eng.compute(
            self._world, self._prev_world, action_str
        )
        reward = reward_info["total_reward"]

        # --- Generate causal explanation ---
        explanation = self._explainer.explain(
            action=action_str,
            prev_state=self._prev_world,
            curr_state=self._world,
            active_events=active_events,
            step=self._state_obj.step_count,
        )

        # --- Advance step counter ---
        self._state_obj.step_count += 1

        # --- Check termination ---
        collapsed = self._check_collapse()
        reached_limit = self._state_obj.step_count >= self._max_steps
        self._done = collapsed or reached_limit

        # --- Build observation ---
        obs_meta = self._build_observation_metadata(
            reward_breakdown=reward_info,
            active_events=active_events,
            explanation=explanation,
        )
        self._trajectory.append(copy.deepcopy(obs_meta))

        # --- If done, include final grader score ---
        if self._done:
            final_score = grade_trajectory(self._task_id, self._trajectory)
            obs_meta["final_score"] = final_score
            obs_meta["collapsed"] = collapsed
            obs_meta["total_steps"] = self._state_obj.step_count

        return Observation(done=self._done, reward=reward, metadata=obs_meta)

    # =================================================================
    # state property
    # =================================================================

    @property
    def state(self) -> State:
        return self._state_obj

    # =================================================================
    # close
    # =================================================================

    def close(self) -> None:
        pass

    # =================================================================
    # Extra public methods (used by inference script)
    # =================================================================

    def get_trajectory(self) -> List[Dict]:
        """Return the full trajectory of observation metadata."""
        return list(self._trajectory)

    def get_valid_actions(self) -> List[str]:
        """Return list of valid action strings."""
        return list(VALID_ACTIONS)

    def get_action_descriptions(self) -> Dict[str, str]:
        """Return action → description mapping."""
        return dict(ACTION_DESCRIPTIONS)

    # =================================================================
    # Internal helpers
    # =================================================================

    def _parse_action(self, action: Any) -> str:
        """Extract the action string from various input formats."""
        action_str = "no_action"

        if isinstance(action, str):
            action_str = action
        elif hasattr(action, "action"):
            action_str = getattr(action, "action", "no_action")
        elif isinstance(action, dict):
            action_str = action.get("action", "no_action")
        elif hasattr(action, "metadata"):
            meta = getattr(action, "metadata", {})
            if isinstance(meta, dict):
                action_str = meta.get("action", "no_action")

        if action_str not in VALID_ACTIONS:
            action_str = "no_action"

        return action_str

    def _check_collapse(self) -> bool:
        """Check if any collapse condition is met."""
        for cond_name, (metric, threshold) in COLLAPSE_CONDITIONS.items():
            val = self._world.get(metric, 50.0)
            if cond_name == "gdp_collapse" and val < threshold:
                return True
            if cond_name == "eco_collapse" and val > threshold:
                return True
            if cond_name == "social_collapse" and val < threshold:
                return True
        return False

    def _build_observation_metadata(
        self,
        reward_breakdown: Optional[Dict] = None,
        active_events: Optional[List[str]] = None,
        explanation: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Assemble the full observation metadata dict."""
        cfg = TASK_CONFIGS[self._task_id]
        meta: Dict[str, Any] = {}

        # All world-state metrics
        for key, val in self._world.items():
            meta[key] = round(val, 4)

        # Temporal context
        meta["step_number"] = self._state_obj.step_count
        meta["max_steps"] = self._max_steps
        meta["last_actions"] = list(self._last_actions)
        meta["active_events"] = active_events or []

        # Task info
        meta["task_id"] = self._task_id
        meta["task_description"] = cfg["description"]

        # Reward breakdown
        if reward_breakdown:
            meta["reward_breakdown"] = reward_breakdown

        # Explainability -- causal reasoning chain
        if explanation:
            meta["explanation"] = explanation

        return meta
