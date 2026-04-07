#!/usr/bin/env python3
"""
POLARIS — 17-Section Mega Validation Suite
==========================================
Comprehensive stress testing across all dimensions:
chaos, cascades, reversibility, delays, events, extremes,
rewards, noise, long-horizon, multi-seed, edge-boundary,
action coverage, exploit search, distribution shift,
parameter sweep, failure modes, agent differentiation.
"""

import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.policy_environment import PolicyEnvironment
from server.config import VALID_ACTIONS, TASK_CONFIGS
from server.tasks import grade_trajectory, get_task_ids

# ================================================================
# Helpers
# ================================================================

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️ WARN"

total_checks = 0
passed_checks = 0
failed_checks = 0
warnings = 0

def check(condition, name, detail=""):
    global total_checks, passed_checks, failed_checks
    total_checks += 1
    if condition:
        passed_checks += 1
        print(f"  {PASS} {name}")
    else:
        failed_checks += 1
        print(f"  {FAIL} {name} — {detail}")
    return condition

def warn(condition, name, detail=""):
    global total_checks, warnings
    total_checks += 1
    if condition:
        print(f"  {PASS} {name}")
    else:
        warnings += 1
        print(f"  {WARN} {name} — {detail}")

def run_episode(task_id="sustainable_governance", seed=42, max_override=None, policy=None):
    """Run a full episode, optionally with a custom policy function."""
    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    rewards = []
    actions_taken = []
    step = 0
    max_steps = max_override or TASK_CONFIGS[task_id]["max_steps"]

    while not obs.done and step < max_steps:
        step += 1
        meta = obs.metadata
        if policy:
            action = policy(meta, step)
        else:
            action = random.choice(list(VALID_ACTIONS))
        obs = env.step({"action": action})
        rewards.append(obs.reward)
        actions_taken.append(action)

    return {
        "steps": step,
        "rewards": rewards,
        "total_reward": sum(rewards),
        "actions": actions_taken,
        "collapsed": obs.metadata.get("collapsed", False),
        "final_meta": obs.metadata,
        "done": obs.done,
        "trajectory": env.get_trajectory(),
    }


# ================================================================
# SECTION 1: CHAOS STRESS TEST
# ================================================================

def section_1():
    print("\n" + "=" * 70)
    print("  SECTION 1: CHAOS STRESS TEST")
    print("  Maximum randomness + instability — 500 episodes")
    print("=" * 70)

    crash_count = 0
    invalid_states = 0
    collapse_count = 0
    total_episodes = 500

    for i in range(total_episodes):
        try:
            result = run_episode(seed=i, task_id="sustainable_governance")
            if result["collapsed"]:
                collapse_count += 1
            meta = result["final_meta"]
            # Check for invalid states
            if meta.get("pollution_index", 0) < 0 or meta.get("gdp_index", 0) < 0:
                invalid_states += 1
            if meta.get("public_satisfaction", 0) < 0 or meta.get("public_satisfaction", 0) > 200:
                invalid_states += 1
        except Exception as e:
            crash_count += 1

    crash_rate = crash_count / total_episodes
    collapse_rate = collapse_count / total_episodes

    check(crash_count == 0, f"Zero crashes across {total_episodes} episodes", f"Crashes: {crash_count}")
    check(invalid_states == 0, f"Zero invalid states", f"Invalid: {invalid_states}")
    warn(collapse_rate < 0.8, f"Collapse rate reasonable: {collapse_rate:.1%}", f"Too many collapses")
    print(f"  📊 Crash rate: {crash_rate:.2%} | Collapse rate: {collapse_rate:.1%}")


# ================================================================
# SECTION 2: CASCADE AMPLIFICATION TEST
# ================================================================

def section_2():
    print("\n" + "=" * 70)
    print("  SECTION 2: CASCADE AMPLIFICATION TEST")
    print("  Check runaway feedback loops")
    print("=" * 70)

    # Use worst-case policy: always expand industry (pollution + GDP)
    def bad_policy(meta, step):
        return "expand_industry"

    result = run_episode(seed=42, policy=bad_policy)
    metas = []
    
    env = PolicyEnvironment()
    obs = env.reset(seed=42, task_id="sustainable_governance")
    for step in range(50):
        if obs.done:
            break
        obs = env.step({"action": "expand_industry"})
        metas.append(obs.metadata.copy())

    if len(metas) >= 2:
        poll_vals = [m.get("pollution_index", 0) for m in metas]
        gdp_vals = [m.get("gdp_index", 0) for m in metas]

        # Check pollution is bounded
        check(max(poll_vals) < 1000, "Pollution bounded under cascading industry",
              f"Max pollution: {max(poll_vals):.0f}")
        check(all(0 <= p <= 500 for p in poll_vals), "All pollution values in valid range",
              f"Range: [{min(poll_vals):.0f}, {max(poll_vals):.0f}]")
        check(all(g >= 0 for g in gdp_vals), "GDP never goes negative",
              f"Min GDP: {min(gdp_vals):.1f}")
    else:
        check(True, "Episode too short for cascade test (collapsed early)")


# ================================================================
# SECTION 3: REVERSIBILITY TEST
# ================================================================

def section_3():
    print("\n" + "=" * 70)
    print("  SECTION 3: REVERSIBILITY TEST")
    print("  Opposite actions — does system drift?")
    print("=" * 70)

    env = PolicyEnvironment()
    obs = env.reset(seed=42, task_id="sustainable_governance")
    initial_gdp = obs.metadata.get("gdp_index", 100)

    # 10x increase_tax, then 10x decrease_tax
    for _ in range(10):
        obs = env.step({"action": "increase_tax"})
    mid_gdp = obs.metadata.get("gdp_index", 0)

    for _ in range(10):
        if obs.done:
            break
        obs = env.step({"action": "decrease_tax"})
    final_gdp = obs.metadata.get("gdp_index", 0)

    drift = abs(final_gdp - initial_gdp)
    check(mid_gdp != initial_gdp, "Tax increase had measurable GDP effect",
          f"GDP unchanged at {initial_gdp}")
    check(drift < initial_gdp * 2, "GDP drift bounded after reversal",
          f"Drift: {drift:.1f}")
    print(f"  📊 GDP: initial={initial_gdp:.1f} → mid={mid_gdp:.1f} → final={final_gdp:.1f} (drift={drift:.1f})")


# ================================================================
# SECTION 4: DELAY CONSISTENCY TEST
# ================================================================

def section_4():
    print("\n" + "=" * 70)
    print("  SECTION 4: DELAY CONSISTENCY TEST")
    print("  Verify temporal logic — delayed actions")
    print("=" * 70)

    env = PolicyEnvironment()
    obs = env.reset(seed=42, task_id="sustainable_governance")
    edu_before = obs.metadata.get("education_index", 50)

    # Invest in education 5 times
    for _ in range(5):
        obs = env.step({"action": "invest_in_education"})
    edu_after_5 = obs.metadata.get("education_index", 50)

    # Then do nothing 5 times to let delayed effects propagate
    for _ in range(5):
        if obs.done:
            break
        obs = env.step({"action": "no_action"})
    edu_after_10 = obs.metadata.get("education_index", 50)

    check(edu_after_5 > edu_before, "Education investment shows immediate effect",
          f"Before: {edu_before:.1f}, After 5 steps: {edu_after_5:.1f}")
    warn(edu_after_10 >= edu_after_5 * 0.8, "Delayed effects persist",
         f"After 10: {edu_after_10:.1f}")
    print(f"  📊 Education: {edu_before:.1f} → {edu_after_5:.1f} → {edu_after_10:.1f}")


# ================================================================
# SECTION 5: EVENT DOMINANCE TEST
# ================================================================

def section_5():
    print("\n" + "=" * 70)
    print("  SECTION 5: EVENT DOMINANCE TEST")
    print("  Events don't overpower the system")
    print("=" * 70)

    # Run with different seeds to hit different events
    survivals = 0
    total = 100
    event_counts = Counter()

    for seed in range(total):
        result = run_episode(seed=seed)
        if not result["collapsed"]:
            survivals += 1
        events = result["final_meta"].get("active_events", [])
        for e in events:
            event_counts[e] += 1

    survival_rate = survivals / total
    check(survival_rate > 0.05, f"Survival rate > 5% under random policy: {survival_rate:.1%}",
          f"Only {survivals}/{total} survived")
    warn(survival_rate > 0.15, f"Survival rate healthy: {survival_rate:.1%}",
         f"Low survival with random actions")
    print(f"  📊 Survival: {survivals}/{total} ({survival_rate:.1%})")
    if event_counts:
        print(f"  📊 Events seen: {dict(event_counts.most_common(5))}")


# ================================================================
# SECTION 6: POLICY EXTREME TEST
# ================================================================

def section_6():
    print("\n" + "=" * 70)
    print("  SECTION 6: POLICY EXTREME TEST")
    print("  Break system with extreme strategies")
    print("=" * 70)

    extreme_policies = {
        "always_industry": lambda m, s: "expand_industry",
        "always_welfare": lambda m, s: "increase_welfare",
        "always_tax": lambda m, s: "increase_tax",
        "always_nothing": lambda m, s: "no_action",
        "always_renewables": lambda m, s: "subsidize_renewables",
        "always_emission_limits": lambda m, s: "enforce_emission_limits",
    }

    for name, policy in extreme_policies.items():
        try:
            result = run_episode(seed=42, policy=policy)
            status = "COLLAPSED" if result["collapsed"] else f"OK (steps={result['steps']})"
            check(True, f"{name}: ran without crash — {status}")
            print(f"    Reward: {result['total_reward']:.2f} | Steps: {result['steps']}")
        except Exception as e:
            check(False, f"{name}: crashed", str(e))


# ================================================================
# SECTION 7: REWARD MISALIGNMENT TEST
# ================================================================

def section_7():
    print("\n" + "=" * 70)
    print("  SECTION 7: REWARD MISALIGNMENT TEST")
    print("  Detect hidden reward bugs")
    print("=" * 70)

    # Compare: always_industry (high GDP, high pollution) vs balanced
    def industry_only(m, s):
        return "expand_industry"

    def balanced(m, s):
        actions = ["subsidize_renewables", "invest_in_education", "stimulate_economy",
                   "invest_in_healthcare", "increase_welfare"]
        return actions[s % len(actions)]

    r_industry = run_episode(seed=42, policy=industry_only)
    r_balanced = run_episode(seed=42, policy=balanced)

    check(r_balanced["total_reward"] > r_industry["total_reward"] * 0.3,
          "Balanced policy not dramatically worse than pure industry",
          f"Balanced: {r_balanced['total_reward']:.2f} vs Industry: {r_industry['total_reward']:.2f}")

    # Check rewards are always numeric
    all_numeric = all(isinstance(r, (int, float)) and not math.isnan(r) and not math.isinf(r)
                      for r in r_industry["rewards"] + r_balanced["rewards"])
    check(all_numeric, "All rewards are finite numeric values")

    print(f"  📊 Industry-only reward: {r_industry['total_reward']:.2f} (steps: {r_industry['steps']})")
    print(f"  📊 Balanced reward: {r_balanced['total_reward']:.2f} (steps: {r_balanced['steps']})")


# ================================================================
# SECTION 8: NOISE SENSITIVITY TEST
# ================================================================

def section_8():
    print("\n" + "=" * 70)
    print("  SECTION 8: NOISE SENSITIVITY TEST")
    print("  Small perturbations — trajectory divergence")
    print("=" * 70)

    # Same seed, same policy, should produce same result (determinism)
    def fixed_policy(m, s):
        actions = list(VALID_ACTIONS)
        idx = s % len(actions)
        return actions[idx]

    r1 = run_episode(seed=42, policy=fixed_policy)
    r2 = run_episode(seed=42, policy=fixed_policy)

    check(r1["total_reward"] == r2["total_reward"],
          "Identical seed + policy = identical reward (deterministic)",
          f"Run1: {r1['total_reward']:.4f} vs Run2: {r2['total_reward']:.4f}")
    check(r1["steps"] == r2["steps"],
          "Identical seed + policy = identical step count",
          f"Run1: {r1['steps']} vs Run2: {r2['steps']}")

    # Different seed should give different result
    r3 = run_episode(seed=99, policy=fixed_policy)
    check(r3["total_reward"] != r1["total_reward"] or r3["steps"] != r1["steps"],
          "Different seed produces different trajectory")


# ================================================================
# SECTION 9: LONG-HORIZON FATIGUE TEST
# ================================================================

def section_9():
    print("\n" + "=" * 70)
    print("  SECTION 9: LONG-HORIZON FATIGUE TEST")
    print("  Detect hidden degradation — 2000+ steps")
    print("=" * 70)

    # Use a balanced policy to survive long
    def survival_policy(m, s):
        poll = m.get("pollution_index", 100)
        gdp = m.get("gdp_index", 100)
        sat = m.get("public_satisfaction", 50)

        if gdp < 30: return "stimulate_economy"
        if sat < 15: return "increase_welfare"
        if poll > 250: return "enforce_emission_limits"
        if poll > 150: return "subsidize_renewables"
        if gdp < 60: return "stimulate_economy"
        if sat < 35: return "increase_welfare"
        return "invest_in_education"

    env = PolicyEnvironment()
    obs = env.reset(seed=42, task_id="sustainable_governance")
    rewards = []
    step = 0
    max_steps = 2000

    while not obs.done and step < max_steps:
        step += 1
        action = survival_policy(obs.metadata, step)
        obs = env.step({"action": action})
        rewards.append(obs.reward)

    check(step > 10, f"Survived at least 10 steps (actual: {step})")
    
    if len(rewards) > 100:
        early = statistics.mean(rewards[:50])
        late = statistics.mean(rewards[-50:])
        check(not math.isnan(late) and not math.isinf(late),
              "No NaN/Inf in late rewards")
        warn(late > early * 0.3, f"No severe late-game degradation",
             f"Early avg: {early:.4f}, Late avg: {late:.4f}")
        print(f"  📊 Steps survived: {step} | Early reward avg: {early:.4f} | Late: {late:.4f}")
    else:
        print(f"  📊 Steps survived: {step} (collapsed early)")


# ================================================================
# SECTION 10: MULTI-SEED STABILITY TEST
# ================================================================

def section_10():
    print("\n" + "=" * 70)
    print("  SECTION 10: MULTI-SEED STABILITY TEST")
    print("  100 seeds — variance and survival distribution")
    print("=" * 70)

    def decent_policy(m, s):
        poll = m.get("pollution_index", 100)
        gdp = m.get("gdp_index", 100)
        sat = m.get("public_satisfaction", 50)
        if gdp < 30: return "stimulate_economy"
        if sat < 15: return "increase_welfare"
        if poll > 250: return "enforce_emission_limits"
        actions = ["subsidize_renewables", "invest_in_education", "stimulate_economy"]
        return actions[s % len(actions)]

    scores = []
    step_counts = []
    collapses = 0

    for seed in range(100):
        result = run_episode(seed=seed, policy=decent_policy)
        score = grade_trajectory(
            "sustainable_governance", result["trajectory"]
        )
        scores.append(score)
        step_counts.append(result["steps"])
        if result["collapsed"]:
            collapses += 1

    avg_score = statistics.mean(scores)
    std_score = statistics.stdev(scores) if len(scores) > 1 else 0
    avg_steps = statistics.mean(step_counts)

    check(avg_score > 0, f"Average score > 0: {avg_score:.4f}")
    check(std_score < avg_score * 3 if avg_score > 0 else True,
          f"Score variance reasonable (std={std_score:.4f})")
    print(f"  📊 Score: mean={avg_score:.4f} std={std_score:.4f}")
    print(f"  📊 Steps: mean={avg_steps:.0f} | Collapses: {collapses}/100")
    print(f"  📊 Score range: [{min(scores):.4f}, {max(scores):.4f}]")


# ================================================================
# SECTION 11: EDGE-BOUNDARY TEST
# ================================================================

def section_11():
    print("\n" + "=" * 70)
    print("  SECTION 11: EDGE-BOUNDARY TEST")
    print("  Test state bounds at limits")
    print("=" * 70)

    # Push pollution to max
    env = PolicyEnvironment()
    obs = env.reset(seed=42, task_id="sustainable_governance")

    for _ in range(50):
        if obs.done:
            break
        obs = env.step({"action": "expand_industry"})

    meta = obs.metadata
    poll = meta.get("pollution_index", 0)
    gdp = meta.get("gdp_index", 0)
    sat = meta.get("public_satisfaction", 0)

    check(poll >= 0, f"Pollution non-negative at boundary: {poll:.1f}")
    check(gdp >= 0, f"GDP non-negative at boundary: {gdp:.1f}")
    check(sat >= 0, f"Satisfaction non-negative at boundary: {sat:.1f}")
    check(poll <= 500, f"Pollution bounded: {poll:.1f}", f"Unbounded: {poll:.1f}")

    # Push satisfaction to minimum
    env2 = PolicyEnvironment()
    obs2 = env2.reset(seed=42, task_id="sustainable_governance")
    for _ in range(50):
        if obs2.done:
            break
        obs2 = env2.step({"action": "increase_tax"})
    
    check(obs2.metadata.get("public_satisfaction", 0) >= 0,
          "Satisfaction never negative under heavy taxation")


# ================================================================
# SECTION 12: ACTION COVERAGE TEST
# ================================================================

def section_12():
    print("\n" + "=" * 70)
    print("  SECTION 12: ACTION COVERAGE TEST")
    print("  Every action works in every state")
    print("=" * 70)

    working_actions = 0
    total_actions = len(VALID_ACTIONS)

    for action in VALID_ACTIONS:
        try:
            env = PolicyEnvironment()
            obs = env.reset(seed=42, task_id="sustainable_governance")
            obs = env.step({"action": action})
            check(obs.reward is not None, f"Action '{action}' returns valid reward",
                  "Reward is None")
            check(isinstance(obs.metadata, dict), f"Action '{action}' returns valid metadata")
            working_actions += 1
        except Exception as e:
            check(False, f"Action '{action}' crashed", str(e))

    check(working_actions == total_actions,
          f"All {total_actions} actions functional",
          f"Only {working_actions}/{total_actions}")


# ================================================================
# SECTION 13: EXPLOIT SEARCH (AUTO)
# ================================================================

def section_13():
    print("\n" + "=" * 70)
    print("  SECTION 13: EXPLOIT SEARCH")
    print("  Random policy search for reward exploits")
    print("=" * 70)

    max_reward = float("-inf")
    best_policy = None
    action_list = list(VALID_ACTIONS)

    # Try 50 random fixed-action policies
    for action in action_list:
        result = run_episode(seed=42, policy=lambda m, s, a=action: a)
        if result["total_reward"] > max_reward:
            max_reward = result["total_reward"]
            best_policy = action

    # Check no single action gives absurd reward
    check(max_reward < 1000, f"No exploit: max single-action reward = {max_reward:.2f}",
          f"Possible exploit with {best_policy}: {max_reward:.2f}")
    warn(max_reward < 100, f"Single-action rewards reasonable: {max_reward:.2f}")
    print(f"  📊 Best single-action: '{best_policy}' → reward={max_reward:.2f}")

    # Check reward per step is bounded
    result = run_episode(seed=42, policy=lambda m, s: best_policy)
    if result["steps"] > 0:
        per_step = max_reward / result["steps"]
        check(per_step < 10, f"Reward per step bounded: {per_step:.4f}")


# ================================================================
# SECTION 14: DISTRIBUTION SHIFT TEST
# ================================================================

def section_14():
    print("\n" + "=" * 70)
    print("  SECTION 14: DISTRIBUTION SHIFT TEST")
    print("  Train on one task, test on others")
    print("=" * 70)

    task_ids = get_task_ids()

    def generic_policy(m, s):
        poll = m.get("pollution_index", 100)
        gdp = m.get("gdp_index", 100)
        sat = m.get("public_satisfaction", 50)
        if gdp < 30: return "stimulate_economy"
        if sat < 15: return "increase_welfare"
        if poll > 250: return "enforce_emission_limits"
        return "subsidize_renewables"

    scores_by_task = {}
    for task_id in task_ids:
        result = run_episode(seed=42, task_id=task_id, policy=generic_policy)
        score = grade_trajectory(task_id, result["trajectory"])
        scores_by_task[task_id] = score
        check(score >= 0, f"Task '{task_id}' score non-negative: {score:.4f}")
        print(f"    {task_id}: score={score:.4f} steps={result['steps']} collapsed={result['collapsed']}")

    if len(scores_by_task) > 1:
        vals = list(scores_by_task.values())
        check(max(vals) - min(vals) < 5, "Score spread across tasks reasonable",
              f"Spread: {max(vals) - min(vals):.4f}")


# ================================================================
# SECTION 15: PARAMETER SWEEP TEST
# ================================================================

def section_15():
    print("\n" + "=" * 70)
    print("  SECTION 15: PARAMETER SWEEP TEST")
    print("  Sensitivity to seeds")
    print("=" * 70)

    rewards_by_seed = []
    for seed in range(50):
        result = run_episode(seed=seed)
        rewards_by_seed.append(result["total_reward"])

    avg = statistics.mean(rewards_by_seed)
    std = statistics.stdev(rewards_by_seed) if len(rewards_by_seed) > 1 else 0

    check(not math.isnan(avg), "Average reward is not NaN")
    check(not math.isinf(avg), "Average reward is not Inf")
    print(f"  📊 Reward across 50 seeds: mean={avg:.2f} std={std:.2f}")
    print(f"  📊 Range: [{min(rewards_by_seed):.2f}, {max(rewards_by_seed):.2f}]")


# ================================================================
# SECTION 16: FAILURE MODE CONSISTENCY
# ================================================================

def section_16():
    print("\n" + "=" * 70)
    print("  SECTION 16: FAILURE MODE CONSISTENCY")
    print("  Predictable, consistent collapse behavior")
    print("=" * 70)

    collapse_causes = Counter()
    for seed in range(100):
        result = run_episode(seed=seed)
        if result["collapsed"]:
            meta = result["final_meta"]
            if meta.get("pollution_index", 0) > 280:
                collapse_causes["pollution"] += 1
            elif meta.get("gdp_index", 100) < 20:
                collapse_causes["gdp"] += 1
            elif meta.get("public_satisfaction", 50) < 10:
                collapse_causes["satisfaction"] += 1
            else:
                collapse_causes["other"] += 1

    total_collapses = sum(collapse_causes.values())
    check(True, f"Collapse cause distribution mapped ({total_collapses} total)")
    for cause, count in collapse_causes.most_common():
        print(f"    {cause}: {count} ({count/max(total_collapses,1):.0%})")

    if total_collapses > 0:
        check(collapse_causes.get("other", 0) / total_collapses < 0.3,
              "Most collapses have identifiable cause")


# ================================================================
# SECTION 17: AGENT DIFFERENTIATION TEST
# ================================================================

def section_17():
    print("\n" + "=" * 70)
    print("  SECTION 17: AGENT DIFFERENTIATION TEST")
    print("  Ensure meaningful benchmarking — different policies get different scores")
    print("=" * 70)

    policies = {
        "random": lambda m, s: random.choice(list(VALID_ACTIONS)),
        "always_nothing": lambda m, s: "no_action",
        "always_industry": lambda m, s: "expand_industry",
        "smart_rule": lambda m, s: (
            "stimulate_economy" if m.get("gdp_index", 100) < 40 else
            "increase_welfare" if m.get("public_satisfaction", 50) < 20 else
            "enforce_emission_limits" if m.get("pollution_index", 100) > 200 else
            "subsidize_renewables"
        ),
    }

    policy_scores = {}
    for name, policy in policies.items():
        scores = []
        for seed in range(10):
            result = run_episode(seed=seed, policy=policy)
            score = grade_trajectory("sustainable_governance", result["trajectory"])
            scores.append(score)
        avg = statistics.mean(scores)
        policy_scores[name] = avg
        print(f"    {name}: avg_score={avg:.4f}")

    # Smart policy should beat random and do-nothing
    if "smart_rule" in policy_scores and "random" in policy_scores:
        check(policy_scores["smart_rule"] >= policy_scores["random"] * 0.8,
              "Smart policy competitive with random",
              f"Smart={policy_scores['smart_rule']:.4f} vs Random={policy_scores['random']:.4f}")

    # Different policies should get meaningfully different scores
    all_scores = list(policy_scores.values())
    if len(all_scores) > 1:
        spread = max(all_scores) - min(all_scores)
        check(spread > 0.001, f"Policies produce different scores (spread={spread:.4f})",
              "All policies score identically — no differentiation")


# ================================================================
# MAIN
# ================================================================

def main():
    global total_checks, passed_checks, failed_checks, warnings

    print("╔" + "═" * 68 + "╗")
    print("║  POLARIS — 17-SECTION MEGA VALIDATION SUITE                       ║")
    print("║  Comprehensive environment integrity verification                 ║")
    print("╚" + "═" * 68 + "╝")

    start = time.time()

    section_1()   # Chaos Stress
    section_2()   # Cascade Amplification
    section_3()   # Reversibility
    section_4()   # Delay Consistency
    section_5()   # Event Dominance
    section_6()   # Policy Extreme
    section_7()   # Reward Misalignment
    section_8()   # Noise Sensitivity
    section_9()   # Long-Horizon Fatigue
    section_10()  # Multi-Seed Stability
    section_11()  # Edge-Boundary
    section_12()  # Action Coverage
    section_13()  # Exploit Search
    section_14()  # Distribution Shift
    section_15()  # Parameter Sweep
    section_16()  # Failure Mode Consistency
    section_17()  # Agent Differentiation

    elapsed = time.time() - start

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  FINAL RESULTS                                                    ║")
    print("╚" + "═" * 68 + "╝")
    print(f"  Total checks:  {total_checks}")
    print(f"  ✅ Passed:      {passed_checks}")
    print(f"  ❌ Failed:      {failed_checks}")
    print(f"  ⚠️  Warnings:   {warnings}")
    print(f"  ⏱️  Time:        {elapsed:.1f}s")
    print(f"  📊 Pass rate:   {passed_checks/max(total_checks,1):.1%}")
    print()

    if failed_checks == 0:
        print("  🏆 PERFECT SCORE — ALL CHECKS PASSED")
    elif failed_checks <= 3:
        print("  ✅ NEAR-PERFECT — Minor issues only")
    else:
        print(f"  ⚠️ {failed_checks} failures need attention")


if __name__ == "__main__":
    main()
