#!/usr/bin/env python3
"""
AI Policy Engine — RL Agent v3 (Final)

REINFORCE with:
  - Curriculum learning: easy → medium → hard with weight transfer
  - Survival reward shaping (danger penalty + collapse penalty)
  - Annealed entropy regularisation + cosine LR decay
  - 1-hidden-layer MLP (21 → 96 → 16) — fast pure-Python backprop
  - γ=0.997 for long-horizon hard task

Zero ML dependencies.
"""

import sys, os, json, time, math, random, copy
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.policy_environment import PolicyEnvironment
from server.tasks import grade_trajectory, get_task_ids
from server.config import VALID_ACTIONS, STATE_BOUNDS, TASK_CONFIGS


# =====================================================================
# Neural Network (1 hidden layer — fast backprop)
# =====================================================================

def _rand_matrix(rows, cols, scale=0.1, rng=None):
    r = rng or random
    bound = scale * math.sqrt(6.0 / (rows + cols))
    return [[r.uniform(-bound, bound) for _ in range(cols)] for _ in range(rows)]

def _zeros(n):
    return [0.0] * n

def _dot(mat, vec):
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]

def _add(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def _relu(x):
    return [max(0, v) for v in x]

def _softmax(logits):
    max_l = max(logits)
    exp_l = [math.exp(min(l - max_l, 80)) for l in logits]
    s = sum(exp_l)
    return [e / s for e in exp_l]


class PolicyNetwork:
    def __init__(self, state_dim, hidden_dim, action_dim, lr=0.001, seed=42):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.lr = lr
        self.rng = random.Random(seed)
        self.W1 = _rand_matrix(hidden_dim, state_dim, 0.15, self.rng)
        self.b1 = _zeros(hidden_dim)
        self.W2 = _rand_matrix(action_dim, hidden_dim, 0.15, self.rng)
        self.b2 = _zeros(action_dim)

    def forward(self, state):
        h = _add(_dot(self.W1, state), self.b1)
        h_act = _relu(h)
        logits = _add(_dot(self.W2, h_act), self.b2)
        probs = _softmax(logits)
        return probs, h_act, logits

    def select_action(self, state):
        probs, h_act, logits = self.forward(state)
        r = self.rng.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r < cum:
                return i, probs, h_act, logits
        return len(probs) - 1, probs, h_act, logits

    def get_entropy(self, probs):
        return -sum(p * math.log(max(p, 1e-10)) for p in probs)

    def update(self, trajectories, entropy_coeff=0.01, lr_mult=1.0, gamma=0.99):
        all_returns = []
        for states, actions, rewards, hiddens, logits_list in trajectories:
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                all_returns.append(G)

        if not all_returns:
            return 0.0

        baseline = sum(all_returns) / len(all_returns)
        std_ret = max(1e-8, (sum((r - baseline)**2 for r in all_returns) / max(len(all_returns)-1, 1))**0.5)

        dW1 = [[0.0]*self.state_dim for _ in range(self.hidden_dim)]
        db1 = _zeros(self.hidden_dim)
        dW2 = [[0.0]*self.hidden_dim for _ in range(self.action_dim)]
        db2 = _zeros(self.action_dim)
        n_samples = 0
        total_entropy = 0.0

        for states, actions, rewards, hiddens, logits_list in trajectories:
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            for t in range(len(states)):
                advantage = (returns[t] - baseline) / std_ret
                advantage = max(-5.0, min(5.0, advantage))
                state, action, h_act, logits = states[t], actions[t], hiddens[t], logits_list[t]

                probs = _softmax(logits)
                total_entropy += self.get_entropy(probs)

                d_logits = list(probs)
                d_logits[action] -= 1.0
                for i in range(self.action_dim):
                    ent_grad = (1.0 + math.log(max(probs[i], 1e-10))) / self.action_dim
                    d_logits[i] = d_logits[i] * (-advantage) + entropy_coeff * ent_grad

                for i in range(self.action_dim):
                    for j in range(self.hidden_dim):
                        dW2[i][j] += d_logits[i] * h_act[j]
                    db2[i] += d_logits[i]

                d_hidden = [0.0] * self.hidden_dim
                for j in range(self.hidden_dim):
                    for i in range(self.action_dim):
                        d_hidden[j] += d_logits[i] * self.W2[i][j]
                    if h_act[j] <= 0:
                        d_hidden[j] = 0.0

                for i in range(self.hidden_dim):
                    for j in range(self.state_dim):
                        dW1[i][j] += d_hidden[i] * state[j]
                    db1[i] += d_hidden[i]
                n_samples += 1

        if n_samples == 0:
            return 0.0

        effective_lr = self.lr * lr_mult
        scale = effective_lr / n_samples
        clip = 0.5

        for i in range(self.hidden_dim):
            for j in range(self.state_dim):
                self.W1[i][j] -= max(-clip, min(clip, scale * dW1[i][j]))
            self.b1[i] -= max(-clip, min(clip, scale * db1[i]))
        for i in range(self.action_dim):
            for j in range(self.hidden_dim):
                self.W2[i][j] -= max(-clip, min(clip, scale * dW2[i][j]))
            self.b2[i] -= max(-clip, min(clip, scale * db2[i]))

        return total_entropy / n_samples

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"W1":self.W1,"b1":self.b1,"W2":self.W2,"b2":self.b2,
                        "dims":[self.state_dim,self.hidden_dim,self.action_dim]}, f)

    def load(self, path):
        with open(path) as f:
            d = json.load(f)
        self.W1, self.b1, self.W2, self.b2 = d["W1"], d["b1"], d["W2"], d["b2"]

    def copy_weights_from(self, other):
        self.W1 = copy.deepcopy(other.W1)
        self.b1 = copy.deepcopy(other.b1)
        self.W2 = copy.deepcopy(other.W2)
        self.b2 = copy.deepcopy(other.b2)


# =====================================================================
# State normalisation + danger detection
# =====================================================================

STATE_KEYS = [
    "pollution_index", "carbon_emission_rate", "renewable_energy_ratio",
    "ecological_stability", "gdp_index", "industrial_output",
    "unemployment_rate", "inflation_rate", "trade_balance",
    "foreign_investment", "public_satisfaction", "healthcare_index",
    "education_index", "inequality_index", "energy_efficiency",
    "transport_efficiency", "tax_rate", "regulation_strength",
    "welfare_spending", "green_subsidies", "interest_rate",
]
ACTION_LIST = sorted(VALID_ACTIONS)


def normalise_state(obs_metadata):
    vec = []
    for key in STATE_KEYS:
        val = obs_metadata.get(key, 0.0)
        lo, hi = STATE_BOUNDS.get(key, (0, 100))
        norm = 2.0 * (val - lo) / (hi - lo) - 1.0 if hi > lo else 0.0
        vec.append(max(-3.0, min(3.0, norm)))
    return vec


def shape_reward(original_reward, meta, collapsed, step, max_steps, prev_sat=None):
    """Survival reward shaping — heavily targets satisfaction (sole collapse vector)."""
    shaped = original_reward
    gdp = meta.get("gdp_index", 100)
    poll = meta.get("pollution_index", 100)
    sat = meta.get("public_satisfaction", 50)

    progress = step / max_steps

    # Survival bonus (escalates with progress — massive late-game bonus)
    shaped += 0.08 * (1.0 + 2.0 * progress)

    # ── SATISFACTION IS THE #1 THREAT ──
    # Exponential danger as satisfaction approaches collapse (< 5)
    if sat < 35:
        # Smooth exponential penalty: mild at 35, severe at 10
        danger = (35 - sat) / 30  # 0.0 at 35, 1.0 at 5
        shaped -= 0.5 * (danger ** 1.5)  # exponential scaling
    if sat < 15:
        shaped -= 1.0  # emergency alarm zone

    # Satisfaction recovery bonus (reward improvement)
    if prev_sat is not None and sat > prev_sat:
        shaped += 0.15 * min(sat - prev_sat, 5.0)  # cap at +0.75

    # Satisfaction maintenance bonus (keeping it above safe zone)
    if sat > 30:
        shaped += 0.05  # small constant reward for being safe

    # ── Secondary dangers (less critical) ──
    if gdp < 30:
        shaped -= 0.15 * (1.0 - gdp / 30)
    if poll > 230:
        shaped -= 0.15 * ((poll - 230) / 60)

    # Collapse penalty (massive, especially for early collapse)
    if collapsed:
        shaped -= 5.0 * (1.0 - progress)

    return shaped


# =====================================================================
# Training
# =====================================================================

def train(task_id, n_episodes, batch_size=10, lr=0.004, hidden_dim=96,
          seed=42, gamma=0.995, entropy_start=0.06, entropy_end=0.005,
          init_policy=None, verbose=True):

    state_dim = len(STATE_KEYS)
    action_dim = len(ACTION_LIST)
    policy = PolicyNetwork(state_dim, hidden_dim, action_dim, lr=lr, seed=seed)

    if init_policy is not None:
        policy.copy_weights_from(init_policy)
        if verbose:
            print(f"  ← Curriculum: weights transferred from previous task")

    max_steps = TASK_CONFIGS[task_id]["max_steps"]
    use_shaping = True

    episode_rewards, episode_scores, episode_steps, episode_collapses = [], [], [], []
    learning_curve = []
    best_score = -1.0
    best_weights = None

    n_phases = 5
    phase_size = n_episodes // n_phases
    phase_action_dists = [defaultdict(int) for _ in range(n_phases)]

    if verbose:
        print(f"\n{'='*72}")
        print(f"  REINFORCE v3 — '{task_id}'")
        print(f"  Eps: {n_episodes}  Batch: {batch_size}  LR: {lr}  Hidden: {hidden_dim}")
        print(f"  Entropy: {entropy_start}→{entropy_end}  γ={gamma}")
        print(f"  Reward shaping: {'ON' if use_shaping else 'OFF'}")
        print(f"  Curriculum: {'YES (transferred)' if init_policy else 'NO (fresh)'}")
        print(f"{'='*72}")

    t_start = time.time()

    for ep_start in range(0, n_episodes, batch_size):
        batch_trajs = []
        progress = ep_start / n_episodes

        entropy_coeff = entropy_start + (entropy_end - entropy_start) * progress
        if progress < 0.05:
            lr_mult = progress / 0.05
        else:
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * (progress - 0.05) / 0.95))

        phase_idx = min(int(progress * n_phases), n_phases - 1)

        for ep_offset in range(batch_size):
            ep = ep_start + ep_offset
            if ep >= n_episodes:
                break

            env = PolicyEnvironment()
            obs = env.reset(seed=seed + ep, task_id=task_id)
            state = normalise_state(obs.metadata)
            prev_sat = obs.metadata.get("public_satisfaction", 50)

            states, actions, rewards, hiddens, logits_list = [], [], [], [], []
            step = 0

            while not obs.done:
                action_idx, probs, h_act, logits = policy.select_action(state)
                action_name = ACTION_LIST[action_idx]
                obs = env.step({"action": action_name})
                next_state = normalise_state(obs.metadata)
                step += 1

                collapsed = obs.metadata.get("collapsed", False)
                cur_sat = obs.metadata.get("public_satisfaction", 50)
                r = shape_reward(obs.reward, obs.metadata, collapsed, step, max_steps, prev_sat) if use_shaping else obs.reward
                prev_sat = cur_sat

                states.append(state)
                actions.append(action_idx)
                rewards.append(r)
                hiddens.append(h_act)
                logits_list.append(logits)

                phase_action_dists[phase_idx][action_name] += 1
                state = next_state

            total_r = sum(rewards)
            score = grade_trajectory(task_id, env.get_trajectory())

            episode_rewards.append(total_r)
            episode_scores.append(score)
            episode_steps.append(len(rewards))
            episode_collapses.append(obs.metadata.get("collapsed", False))

            batch_trajs.append((states, actions, rewards, hiddens, logits_list))

            if score > best_score:
                best_score = score
                best_weights = {
                    "W1": copy.deepcopy(policy.W1), "b1": copy.deepcopy(policy.b1),
                    "W2": copy.deepcopy(policy.W2), "b2": copy.deepcopy(policy.b2),
                }

        if batch_trajs:
            policy.update(batch_trajs, entropy_coeff=entropy_coeff,
                         lr_mult=lr_mult, gamma=gamma)

        batch_end = min(ep_start + batch_size, n_episodes)
        recent = slice(max(0, batch_end - batch_size), batch_end)
        rc = episode_rewards[recent]
        sc = episode_scores[recent]
        stc = episode_steps[recent]
        cc = episode_collapses[recent]

        learning_curve.append({
            "episode": batch_end,
            "avg_reward": round(sum(rc)/max(len(rc),1), 4),
            "avg_score": round(sum(sc)/max(len(sc),1), 4),
            "avg_steps": round(sum(stc)/max(len(stc),1), 1),
            "collapse_rate": round(sum(cc)/max(len(cc),1), 2),
            "best_score": round(best_score, 4),
            "entropy_coeff": round(entropy_coeff, 4),
            "lr_mult": round(lr_mult, 4),
        })

        if verbose and (batch_end % 200 == 0 or batch_end == n_episodes):
            window = min(200, batch_end)
            w = episode_scores[-window:]
            wr = episode_rewards[-window:]
            wc = episode_collapses[-window:]
            ws = episode_steps[-window:]
            elapsed = time.time() - t_start
            print(f"  Ep {batch_end:5d}/{n_episodes}  "
                  f"r={sum(wr)/len(wr):+7.2f}  score={sum(w)/len(w):.4f}  "
                  f"steps={sum(ws)/len(ws):5.1f}  collapse={sum(wc)/len(wc):.0%}  "
                  f"best={best_score:.4f}  [{elapsed:.1f}s]")

    total_time = time.time() - t_start

    if best_weights:
        policy.W1, policy.b1 = best_weights["W1"], best_weights["b1"]
        policy.W2, policy.b2 = best_weights["W2"], best_weights["b2"]

    action_evolution = {}
    for phase in range(n_phases):
        total = sum(phase_action_dists[phase].values())
        if total > 0:
            dist = {a: round(c/total, 4) for a, c in
                    sorted(phase_action_dists[phase].items(), key=lambda x: -x[1])}
            action_evolution[f"phase_{phase+1}_ep_{phase*phase_size}-{(phase+1)*phase_size}"] = dist

    if verbose:
        print(f"\n  Done in {total_time:.1f}s  Best: {best_score:.4f}")

    return policy, {
        "learning_curve": learning_curve,
        "action_evolution": action_evolution,
        "best_score": best_score,
        "total_time": round(total_time, 1),
        "task_id": task_id,
        "config": {"n_episodes": n_episodes, "batch_size": batch_size, "lr": lr,
                   "hidden_dim": hidden_dim, "entropy_start": entropy_start,
                   "entropy_end": entropy_end, "gamma": gamma,
                   "reward_shaping": True, "curriculum": init_policy is not None},
    }


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_policy(policy, task_id, n_eval=50, seed_base=10000):
    scores, rewards, steps_list = [], [], []
    collapses = 0
    action_counts = defaultdict(int)

    for i in range(n_eval):
        env = PolicyEnvironment()
        obs = env.reset(seed=seed_base + i, task_id=task_id)
        state = normalise_state(obs.metadata)
        total_r, step = 0.0, 0
        eval_rng = random.Random(seed_base + i + 99999)

        while not obs.done:
            probs, _, _ = policy.forward(state)
            logits_t = [math.log(max(p, 1e-10)) / 0.6 for p in probs]
            probs_t = _softmax(logits_t)
            r_val = eval_rng.random()
            cum, action_idx = 0.0, len(probs_t) - 1
            for idx, p in enumerate(probs_t):
                cum += p
                if r_val < cum:
                    action_idx = idx
                    break
            action_name = ACTION_LIST[action_idx]
            action_counts[action_name] += 1
            obs = env.step({"action": action_name})
            total_r += obs.reward
            state = normalise_state(obs.metadata)
            step += 1

        score = grade_trajectory(task_id, env.get_trajectory())
        scores.append(score)
        rewards.append(total_r)
        steps_list.append(step)
        if obs.metadata.get("collapsed"): collapses += 1

    total_acts = sum(action_counts.values())
    top = sorted(action_counts.items(), key=lambda x: -x[1])[:5]
    top_actions = {a: round(c/total_acts, 3) for a, c in top}

    return {
        "avg_score": round(sum(scores)/len(scores), 4),
        "avg_reward": round(sum(rewards)/len(rewards), 4),
        "avg_steps": round(sum(steps_list)/len(steps_list), 1),
        "collapse_rate": round(collapses/n_eval, 2),
        "survival_rate": round(1.0 - collapses/n_eval, 2),
        "best_score": round(max(scores), 4),
        "worst_score": round(min(scores), 4),
        "scores": [round(s, 4) for s in scores],
        "top_actions": top_actions,
    }

def evaluate_random(task_id, n_eval=50, seed_base=10000):
    rng = random.Random(42)
    scores, collapses = [], 0
    for i in range(n_eval):
        env = PolicyEnvironment()
        obs = env.reset(seed=seed_base + i, task_id=task_id)
        while not obs.done:
            obs = env.step({"action": rng.choice(ACTION_LIST)})
        scores.append(grade_trajectory(task_id, env.get_trajectory()))
        if obs.metadata.get("collapsed"): collapses += 1
    return {"avg_score": round(sum(scores)/len(scores), 4), "collapse_rate": round(collapses/n_eval, 2)}

def evaluate_heuristic(task_id, n_eval=50, seed_base=10000):
    cycle = ["subsidize_renewables","invest_in_education","increase_welfare",
             "stimulate_economy","invest_in_healthcare","incentivize_clean_tech",
             "enforce_emission_limits","increase_welfare"]
    scores, collapses = [], 0
    for i in range(n_eval):
        env = PolicyEnvironment()
        obs = env.reset(seed=seed_base + i, task_id=task_id)
        s = 0
        while not obs.done:
            obs = env.step({"action": cycle[s % len(cycle)]})
            s += 1
        scores.append(grade_trajectory(task_id, env.get_trajectory()))
        if obs.metadata.get("collapsed"): collapses += 1
    return {"avg_score": round(sum(scores)/len(scores), 4), "collapse_rate": round(collapses/n_eval, 2)}


# =====================================================================
# Main — Curriculum: Easy -> Medium -> Hard + Extreme evaluation
# =====================================================================

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    all_results = {}

    curriculum = [
        ("environmental_recovery", {
            "n_episodes": 3000, "lr": 0.005, "gamma": 0.99,
            "entropy_start": 0.06, "entropy_end": 0.005, "hidden_dim": 96,
        }),
        ("balanced_economy", {
            "n_episodes": 2500, "lr": 0.004, "gamma": 0.995,
            "entropy_start": 0.07, "entropy_end": 0.008, "hidden_dim": 96,
        }),
        ("sustainable_governance", {
            "n_episodes": 5000, "lr": 0.003, "gamma": 0.997,
            "entropy_start": 0.08, "entropy_end": 0.01, "hidden_dim": 96,
        }),
    ]

    print(f"\n{'='*72}")
    print(f"  CURRICULUM LEARNING PIPELINE (v3 — Dual Regime)")
    print(f"  Stage 1: Environmental Recovery (easy)")
    print(f"  Stage 2: Balanced Economy (medium)")
    print(f"  Stage 3: Sustainable Governance (hard, calibrated)")
    print(f"  Eval:    Sustainable Governance Extreme (structural collapse)")
    print(f"  Total training: {sum(c[1]['n_episodes'] for c in curriculum)} episodes")
    print(f"{'='*72}")

    prev_policy = None

    for task_id, cfg in curriculum:
        policy, train_data = train(
            task_id=task_id,
            n_episodes=cfg["n_episodes"],
            batch_size=10,
            lr=cfg["lr"],
            hidden_dim=cfg["hidden_dim"],
            seed=42,
            gamma=cfg["gamma"],
            entropy_start=cfg["entropy_start"],
            entropy_end=cfg["entropy_end"],
            init_policy=prev_policy,
        )

        policy.save(f"outputs/policy_{task_id}.json")

        print(f"\n  Evaluating '{task_id}' (50 episodes)...")
        trained_eval = evaluate_policy(policy, task_id)
        random_eval = evaluate_random(task_id)
        heuristic_eval = evaluate_heuristic(task_id)

        surv = trained_eval['survival_rate']
        print(f"\n  {'Method':<18s} {'Score':>7s} {'Best':>7s} {'Collapse':>9s} {'Survive':>8s}")
        print(f"  {'-'*52}")
        print(f"  {'RL Agent':<18s} {trained_eval['avg_score']:7.4f} {trained_eval['best_score']:7.4f} {trained_eval['collapse_rate']:8.0%} {surv:7.0%}")
        print(f"  {'Random':<18s} {random_eval['avg_score']:7.4f} {'':>7s} {random_eval['collapse_rate']:8.0%}")
        print(f"  {'Heuristic':<18s} {heuristic_eval['avg_score']:7.4f} {'':>7s} {heuristic_eval['collapse_rate']:8.0%}")

        imp = trained_eval["avg_score"] - random_eval["avg_score"]
        print(f"  vs Random: {imp:+.4f} ({imp/max(random_eval['avg_score'],0.001)*100:+.1f}%)")

        all_results[task_id] = {
            "training": {
                "episodes": cfg["n_episodes"],
                "best_score": train_data["best_score"],
                "time_seconds": train_data["total_time"],
                "learning_curve": train_data["learning_curve"],
                "action_evolution": train_data["action_evolution"],
                "config": train_data["config"],
            },
            "evaluation": {
                "trained": trained_eval,
                "random": random_eval,
                "heuristic": heuristic_eval,
            },
        }

        prev_policy = policy

    # ── Extreme regime evaluation (no training — structural collapse demo) ──
    print(f"\n{'='*72}")
    print(f"  EXTREME REGIME EVALUATION")
    print(f"  Using trained hard-task policy on full-intensity events")
    print(f"{'='*72}")

    extreme_id = "sustainable_governance_extreme"
    print(f"\n  Evaluating '{extreme_id}' (50 episodes)...")
    extreme_trained = evaluate_policy(policy, extreme_id)
    extreme_random = evaluate_random(extreme_id)
    extreme_heuristic = evaluate_heuristic(extreme_id)

    print(f"\n  {'Method':<18s} {'Score':>7s} {'Best':>7s} {'Collapse':>9s} {'Survive':>8s}")
    print(f"  {'-'*52}")
    print(f"  {'RL Agent':<18s} {extreme_trained['avg_score']:7.4f} {extreme_trained['best_score']:7.4f} {extreme_trained['collapse_rate']:8.0%} {extreme_trained['survival_rate']:7.0%}")
    print(f"  {'Random':<18s} {extreme_random['avg_score']:7.4f} {'':>7s} {extreme_random['collapse_rate']:8.0%}")
    print(f"  {'Heuristic':<18s} {extreme_heuristic['avg_score']:7.4f} {'':>7s} {extreme_heuristic['collapse_rate']:8.0%}")

    all_results[extreme_id] = {
        "training": {
            "episodes": 0,
            "best_score": extreme_trained["best_score"],
            "time_seconds": 0,
            "learning_curve": [],
            "action_evolution": {},
            "config": {"note": "No training — uses hard-task policy for structural collapse analysis"},
        },
        "evaluation": {
            "trained": extreme_trained,
            "random": extreme_random,
            "heuristic": extreme_heuristic,
        },
    }

    with open("outputs/rl_training_report.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  FINAL RESULTS — DUAL REGIME ANALYSIS")
    print(f"{'='*72}")
    print(f"\n  {'Task':<35s} {'RL':>7s} {'Rand':>7s} {'Best':>7s} {'Surv%':>7s}")
    print(f"  {'-'*65}")
    for tid, data in all_results.items():
        e = data["evaluation"]
        t = e["trained"]
        marker = " *" if "extreme" in tid else ""
        print(f"  {tid:<35s} {t['avg_score']:7.4f} {e['random']['avg_score']:7.4f} "
              f"{t['best_score']:7.4f} {t['survival_rate']:6.0%}{marker}")
    total_eps = sum(d["training"]["episodes"] for d in all_results.values())
    total_time = sum(d["training"]["time_seconds"] for d in all_results.values())
    print(f"\n  Total: {total_eps} episodes in {total_time:.0f}s")
    print(f"  * = structural collapse regime (no training, eval only)")
    print(f"\n  Key finding: Social stability acts as the primary failure")
    print(f"  bottleneck, dominating economic and environmental factors")
    print(f"  under stochastic pressure.")
    print(f"{'='*72}")

