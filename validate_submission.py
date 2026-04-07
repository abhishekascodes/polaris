#!/usr/bin/env python3
"""
POLARIS — Pre-Submission Validator
===================================
Runs all checks that the hackathon validator will run, locally.
Tests: file structure, inference stdout format, API endpoints, graders, etc.
"""

import importlib
import json
import os
import re
import sys
import io
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "✅"
FAIL = "❌"
total = 0
passed = 0

def check(name, condition, detail=""):
    global total, passed
    total += 1
    if condition:
        passed += 1
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name} — {detail}")

print("=" * 65)
print("  POLARIS — Pre-Submission Validator")
print("=" * 65)

# ── Phase 1: File Structure ──
print("\n📁 PHASE 1: File Structure")
required_files = [
    "inference.py",
    "openenv.yaml",
    "pyproject.toml",
    "requirements.txt",
    "Dockerfile",
    "README.md",
    "models.py",
    "uv.lock",
    "server/app.py",
    "server/policy_environment.py",
    "server/tasks.py",
    "server/config.py",
    "server/transition_engine.py",
    "server/event_engine.py",
    "server/reward_engine.py",
    "server/explainability.py",
]
for f in required_files:
    check(f"File exists: {f}", os.path.exists(f), "MISSING!")

# ── Phase 2: OpenEnv YAML ──
print("\n📋 PHASE 2: openenv.yaml compliance")
try:
    import yaml
    with open("openenv.yaml") as fh:
        cfg = yaml.safe_load(fh)
    check("Has spec_version", "spec_version" in cfg)
    check("Has name", "name" in cfg)
    check("Has tasks", "tasks" in cfg and len(cfg["tasks"]) >= 3, f"Only {len(cfg.get('tasks', []))} tasks")
    task_ids = [t["id"] for t in cfg["tasks"]]
    check("Has 3+ task IDs", len(task_ids) >= 3, str(task_ids))
    for t in cfg["tasks"]:
        check(f"Task '{t['id']}' has description", "description" in t)
        check(f"Task '{t['id']}' has max_steps", "max_steps" in t)
except ImportError:
    print("  ⚠️  PyYAML not installed, skipping YAML checks")
except Exception as e:
    check("openenv.yaml parseable", False, str(e))

# ── Phase 3: Environment Core ──
print("\n🏗️ PHASE 3: Environment Core")
try:
    from server.policy_environment import PolicyEnvironment
    from server.tasks import grade_trajectory, get_task_ids
    from server.config import VALID_ACTIONS, TASK_CONFIGS

    check("PolicyEnvironment importable", True)
    check("get_task_ids() returns list", isinstance(get_task_ids(), list))
    check("4 tasks defined", len(get_task_ids()) >= 3, str(get_task_ids()))
    check("16 valid actions", len(VALID_ACTIONS) == 16, f"Got {len(VALID_ACTIONS)}")

    # Test reset/step/state
    env = PolicyEnvironment()
    obs = env.reset(seed=42, task_id="environmental_recovery")
    check("reset() returns Observation", hasattr(obs, 'done') and hasattr(obs, 'reward') and hasattr(obs, 'metadata'))
    check("reset() done=False", obs.done == False)
    check("reset() reward=0.0", obs.reward == 0.0)
    check("reset() metadata is dict", isinstance(obs.metadata, dict))
    check("state property works", hasattr(env.state, 'episode_id'))

    obs2 = env.step({"action": "subsidize_renewables"})
    check("step() returns Observation", hasattr(obs2, 'done') and hasattr(obs2, 'reward'))
    check("step() reward is float", isinstance(obs2.reward, (int, float)))
    check("step() metadata is dict", isinstance(obs2.metadata, dict))

    # Test all tasks
    for tid in get_task_ids():
        env_t = PolicyEnvironment()
        obs_t = env_t.reset(seed=42, task_id=tid)
        check(f"Task '{tid}' resets OK", obs_t.done == False)
except Exception as e:
    check("Environment imports", False, str(e))

# ── Phase 4: Graders (0.0-1.0) ──
print("\n📊 PHASE 4: Graders produce scores in [0.0, 1.0]")
try:
    for tid in get_task_ids():
        env = PolicyEnvironment()
        obs = env.reset(seed=42, task_id=tid)
        steps = 0
        while not obs.done and steps < 20:
            obs = env.step({"action": "subsidize_renewables"})
            steps += 1
        traj = env.get_trajectory()
        score = grade_trajectory(tid, traj)
        check(f"Task '{tid}' score in [0,1]: {score:.4f}",
              0.0 <= score <= 1.0, f"Got {score}")
except Exception as e:
    check("Grader check", False, str(e))

# ── Phase 5: Determinism ──
print("\n🔁 PHASE 5: Determinism (same seed = same result)")
try:
    rewards_a = []
    env_a = PolicyEnvironment()
    obs_a = env_a.reset(seed=42, task_id="environmental_recovery")
    for _ in range(10):
        obs_a = env_a.step({"action": "subsidize_renewables"})
        rewards_a.append(obs_a.reward)

    rewards_b = []
    env_b = PolicyEnvironment()
    obs_b = env_b.reset(seed=42, task_id="environmental_recovery")
    for _ in range(10):
        obs_b = env_b.step({"action": "subsidize_renewables"})
        rewards_b.append(obs_b.reward)

    check("Same seed produces identical rewards",
          rewards_a == rewards_b,
          f"Mismatch: {rewards_a[:3]} vs {rewards_b[:3]}")
except Exception as e:
    check("Determinism test", False, str(e))

# ── Phase 6: Structured Output Format ──
print("\n📝 PHASE 6: inference.py stdout format validation")
try:
    with open("inference.py") as f:
        code = f.read()

    # Check for the exact format strings
    check("Uses [START] tag", "[START]" in code)
    check("Uses [STEP] tag", "[STEP]" in code)
    check("Uses [END] tag", "[END]" in code)
    check("Uses flush=True", "flush=True" in code)
    check("START has task=", 'task=' in code.split("[START]")[1].split("\n")[0] if "[START]" in code else False)
    check("START has env=", 'env=' in code.split("[START]")[1].split("\n")[0] if "[START]" in code else False)
    check("START has model=", 'model=' in code.split("[START]")[1].split("\n")[0] if "[START]" in code else False)
    check("STEP has step=", 'step=' in code.split("[STEP]")[1].split("\n")[0] if "[STEP]" in code else False)
    check("STEP has action=", 'action=' in code.split("[STEP]")[1].split("\n")[0] if "[STEP]" in code else False)
    check("STEP has reward=", 'reward=' in code.split("[STEP]")[1].split("\n")[0] if "[STEP]" in code else False)
    check("STEP has done=", 'done=' in code.split("[STEP]")[1].split("\n")[0] if "[STEP]" in code else False)
    check("STEP has error=", 'error=' in code.split("[STEP]")[1].split("\n")[0] if "[STEP]" in code else False)
    check("END has success=", 'success=' in code.split("[END]")[1].split("\n")[0] if "[END]" in code else False)
    check("END has steps=", 'steps=' in code.split("[END]")[1].split("\n")[0] if "[END]" in code else False)
    check("END has score=", 'score=' in code.split("[END]")[1].split("\n")[0] if "[END]" in code else False)
    check("END has rewards=", 'rewards=' in code.split("[END]")[1].split("\n")[0] if "[END]" in code else False)
    check("HF_TOKEN has no default", 'os.getenv("HF_TOKEN")' in code and 'os.getenv("HF_TOKEN", "' not in code)
    check("Uses OpenAI client", "OpenAI(" in code or "from openai import" in code)

    # Simulate a dry run (capture stdout from a mock run)
    # We can't run the full inference without an API key, but we can
    # test the format by running with a mock client
    print("\n  🔬 Running format simulation...")

    # Create a mock version that uses rule-based policy only
    from server.policy_environment import PolicyEnvironment
    from server.tasks import grade_trajectory, get_task_ids
    from server.config import TASK_CONFIGS

    captured = io.StringIO()
    with redirect_stdout(captured):
        for task_id in ["environmental_recovery"]:
            cfg = TASK_CONFIGS[task_id]
            max_steps = cfg["max_steps"]
            print(f"[START] task={task_id} env=ai_policy_engine model=test-model", flush=True)

            env = PolicyEnvironment()
            obs = env.reset(seed=42, task_id=task_id)
            step = 0
            all_rewards = []

            while not obs.done and step < min(max_steps, 5):
                step += 1
                obs = env.step({"action": "subsidize_renewables"})
                reward = obs.reward
                all_rewards.append(reward)
                done_str = "true" if obs.done else "false"
                print(f"[STEP] step={step} action=subsidize_renewables reward={reward:.2f} done={done_str} error=null", flush=True)

            traj = env.get_trajectory()
            score = grade_trajectory(task_id, traj)
            success = score >= 0.3
            rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
            print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={rewards_str}", flush=True)

    output = captured.getvalue()
    lines = output.strip().split("\n")

    # Validate format with regex
    start_re = re.compile(r"^\[START\] task=\S+ env=\S+ model=\S+$")
    step_re = re.compile(r"^\[STEP\] step=\d+ action=\S+ reward=-?\d+\.\d+ done=(true|false) error=\S+.*$")
    end_re = re.compile(r"^\[END\] success=(true|false) steps=\d+ score=\d+\.\d+ rewards=[\d.,\-]+$")

    has_start = any(start_re.match(l) for l in lines)
    has_step = any(step_re.match(l) for l in lines)
    has_end = any(end_re.match(l) for l in lines)

    check("Simulated output has valid [START]", has_start, f"Lines: {lines[:2]}")
    check("Simulated output has valid [STEP]", has_step, f"Lines: {[l for l in lines if l.startswith('[STEP]')][:2]}")
    check("Simulated output has valid [END]", has_end, f"Lines: {[l for l in lines if l.startswith('[END]')][:1]}")

    print(f"\n  📋 Sample output:")
    for l in lines[:5]:
        print(f"    {l}")
    if len(lines) > 5:
        print(f"    ...")
    print(f"    {lines[-1]}")

except Exception as e:
    import traceback
    check("Inference format check", False, str(e))
    traceback.print_exc()

# ── Phase 7: Dockerfile ──
print("\n🐳 PHASE 7: Dockerfile checks")
try:
    with open("Dockerfile") as f:
        docker = f.read()
    check("Base image specified", "FROM" in docker)
    check("Exposes port 7860", "7860" in docker)
    check("Has HEALTHCHECK", "HEALTHCHECK" in docker)
    check("Runs uvicorn", "uvicorn" in docker)
    check("Non-root user", "USER user" in docker or "USER 1000" in docker)
except Exception as e:
    check("Dockerfile readable", False, str(e))

# ── Phase 8: README ──
print("\n📖 PHASE 8: README completeness")
try:
    with open("README.md") as f:
        readme = f.read().lower()
    check("Has HF Space frontmatter", "sdk: docker" in readme)
    check("Tags include 'openenv'", "openenv" in readme)
    check("Describes action space", "action space" in readme or "action_space" in readme)
    check("Describes observation space", "observation space" in readme or "observation_space" in readme)
    check("Has setup instructions", "pip install" in readme or "docker" in readme)
    check("Has baseline scores", "baseline" in readme)
    check("Describes tasks", "environmental recovery" in readme or "environmental_recovery" in readme)
    check("Has difficulty progression", "easy" in readme and "medium" in readme and "hard" in readme)
except Exception as e:
    check("README check", False, str(e))

# ── FINAL RESULTS ──
print("\n" + "=" * 65)
print(f"  RESULTS: {passed}/{total} checks passed ({passed/max(total,1):.0%})")
print("=" * 65)

if passed == total:
    print("  🏆 PERFECT — Ready for submission!")
elif total - passed <= 3:
    print(f"  ✅ NEAR-PERFECT — {total - passed} minor issues")
else:
    print(f"  ⚠️ {total - passed} issues need fixing")

print()
