#!/usr/bin/env python3
"""
POLARIS — Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME (optional) for from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

STDOUT FORMAT
- The script emits exactly three line types to stdout:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

# -- Local imports --
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.policy_environment import PolicyEnvironment
from server.tasks import grade_trajectory, get_task_ids
from server.config import VALID_ACTIONS, ACTION_DESCRIPTIONS, TASK_CONFIGS


# ====================================================================
# Configuration
# ====================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "ai_policy_engine"
SEED = 42
SUCCESS_THRESHOLD = 0.3


# ====================================================================
# Policy Reasoning Engine (rule-based pre-filter)
# ====================================================================

class PolicyReasoner:
    @staticmethod
    def analyse(meta: Dict) -> Tuple[Optional[str], str, List[str]]:
        poll = meta.get("pollution_index", 100)
        gdp = meta.get("gdp_index", 100)
        sat = meta.get("public_satisfaction", 50)
        unemp = meta.get("unemployment_rate", 10)
        hc = meta.get("healthcare_index", 50)
        edu = meta.get("education_index", 50)
        rer = meta.get("renewable_energy_ratio", 0.15)
        ineq = meta.get("inequality_index", 40)
        events = meta.get("active_events", [])

        reasons = []
        shortlist = []
        override = None

        if gdp < 30:
            return "stimulate_economy", "GDP critical", ["stimulate_economy"]
        if sat < 15:
            return "increase_welfare", "Satisfaction critical", ["increase_welfare"]
        if poll > 260:
            return "enforce_emission_limits", "Pollution critical", ["enforce_emission_limits"]

        if poll > 180:
            shortlist.extend(["restrict_polluting_industries", "enforce_emission_limits", "implement_carbon_tax"])
        elif poll > 120:
            shortlist.extend(["subsidize_renewables", "enforce_emission_limits"])

        if gdp < 50:
            shortlist.extend(["stimulate_economy", "decrease_tax", "reduce_interest_rates"])
        elif gdp < 70:
            shortlist.append("stimulate_economy")

        if sat < 30:
            shortlist.extend(["increase_welfare", "invest_in_healthcare"])
        elif sat < 45:
            shortlist.append("increase_welfare")

        if unemp > 20:
            shortlist.extend(["expand_industry", "stimulate_economy"])
        if hc < 35:
            shortlist.append("invest_in_healthcare")
        if rer < 0.25 and poll > 100:
            shortlist.extend(["subsidize_renewables", "incentivize_clean_tech"])
        if edu < 40:
            shortlist.append("invest_in_education")
        if ineq > 55:
            if "increase_welfare" not in shortlist:
                shortlist.append("increase_welfare")

        for event in events:
            if event == "pandemic":
                shortlist.insert(0, "invest_in_healthcare")
            elif event == "economic_recession":
                shortlist.insert(0, "stimulate_economy")
            elif event == "climate_crisis":
                shortlist.insert(0, "enforce_emission_limits")
            elif event == "public_protest":
                shortlist.insert(0, "increase_welfare")

        if not shortlist:
            if rer < 0.3:
                shortlist.append("subsidize_renewables")
            if edu < 60:
                shortlist.append("invest_in_education")
            shortlist.append("incentivize_clean_tech")

        seen = set()
        unique = []
        for a in shortlist:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        shortlist = unique[:5]

        return override, "; ".join(reasons) if reasons else "Stable", shortlist


# ====================================================================
# System prompt
# ====================================================================

SYSTEM_PROMPT = """You are an expert AI policy advisor governing a simulated nation.
Each turn you must choose EXACTLY ONE policy action.

AVAILABLE ACTIONS:
{actions}

RULES:
1. Respond with ONLY the action name. No explanation, no formatting, no quotes.
2. Consider delayed effects: education and renewable investments pay off later.
3. Avoid oscillating between opposite actions.
4. Prevent collapse: GDP > 15, pollution < 290, satisfaction > 5.
5. Balance economy, environment, and society.

Respond with EXACTLY one action name from the list above. Nothing else."""

ACTION_LIST = "\n".join(
    f"  - {name}: {desc}" for name, desc in ACTION_DESCRIPTIONS.items()
)


# ====================================================================
# Observation formatting
# ====================================================================

def format_observation(meta: Dict, step: int, max_steps: int,
                       reasoning: str, shortlist: List[str]) -> str:
    lines = [
        f"--- STEP {step}/{max_steps} ---",
        f"Task: {meta.get('task_description', 'N/A')}",
        "",
        f"Pollution: {meta.get('pollution_index', 0):.1f}/300",
        f"GDP: {meta.get('gdp_index', 0):.1f}/200",
        f"Satisfaction: {meta.get('public_satisfaction', 0):.1f}/100",
        f"Unemployment: {meta.get('unemployment_rate', 0):.1f}%",
        f"Healthcare: {meta.get('healthcare_index', 0):.1f}/100",
        f"Education: {meta.get('education_index', 0):.1f}/100",
        f"Renewables: {meta.get('renewable_energy_ratio', 0):.1%}",
        "",
        f"ANALYSIS: {reasoning}",
        f"RECOMMENDED: {', '.join(shortlist)}",
        "Choose your action:",
    ]
    return "\n".join(lines)


# ====================================================================
# LLM agent
# ====================================================================

def get_llm_action(client: OpenAI, observation_text: str, model: str) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(actions=ACTION_LIST)},
                {"role": "user", "content": observation_text},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        raw = response.choices[0].message.content.strip().lower()
        raw = raw.strip("'\"` \n")

        if raw in VALID_ACTIONS:
            return raw
        for action in VALID_ACTIONS:
            if action in raw:
                return action
        return "no_action"
    except Exception:
        return "no_action"


# ====================================================================
# Task runner — EXACT [START]/[STEP]/[END] structured output
# ====================================================================

def run_task(client: OpenAI, task_id: str, seed: int = SEED) -> Dict:
    """Run a single task with exact structured stdout format."""
    cfg = TASK_CONFIGS[task_id]
    max_steps = cfg["max_steps"]
    reasoner = PolicyReasoner()

    # ── [START] ──
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    total_reward = 0.0
    step = 0
    all_rewards: List[float] = []
    last_error = None

    try:
        while not obs.done and step < max_steps:
            step += 1
            meta = obs.metadata

            # Policy reasoning
            override, reasoning, shortlist = reasoner.analyse(meta)

            if override:
                action = override
            else:
                obs_text = format_observation(meta, step, max_steps, reasoning, shortlist)
                action = get_llm_action(client, obs_text, MODEL_NAME)

            # Execute
            obs = env.step({"action": action})
            reward = obs.reward
            total_reward += reward
            all_rewards.append(reward)

            # Error from environment (if any)
            error_str = obs.metadata.get("last_action_error", None)
            error_out = str(error_str) if error_str else "null"

            # Done flag
            done_str = "true" if obs.done else "false"

            # ── [STEP] ──
            print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_out}", flush=True)

    except Exception as e:
        last_error = str(e)

    # Grade
    trajectory = env.get_trajectory()
    score = grade_trajectory(task_id, trajectory)
    success = score >= SUCCESS_THRESHOLD and not obs.metadata.get("collapsed", False)

    # Format rewards list
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)

    # ── [END] ──
    print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={rewards_str}", flush=True)

    return {
        "task_id": task_id,
        "score": score,
        "steps": step,
        "success": success,
    }


# ====================================================================
# Main
# ====================================================================

def main() -> None:
    """Entry point — runs all tasks with structured output."""
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    for task_id in get_task_ids():
        run_task(client, task_id, seed=SEED)


if __name__ == "__main__":
    main()
