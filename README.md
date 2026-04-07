---
title: AI Policy Engine
emoji: "\U0001F3DB"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - multi-objective
  - governance-simulation
  - policy-optimization
  - explainable-ai
pinned: true
---

# AI Policy Engine

**A multi-objective, event-driven governance simulation environment for reinforcement learning agents.**

This environment enables benchmarking of LLM-based policy agents under multi-objective, temporally-dependent decision constraints with full causal explainability.

**This environment evaluates whether an AI can govern -- not just predict.**

---

## Why This Matters

Governments make decisions every day that affect millions of people. Every policy choice involves trade-offs that play out over months and years: investing in renewable energy hurts GDP today but saves the climate tomorrow. Cutting taxes boosts growth but deepens inequality. Increasing welfare spending improves satisfaction but strains the budget.

No existing AI benchmark captures this reality. Game environments test reflexes. Text benchmarks test knowledge. Neither tests what matters most: **the ability to reason about trade-offs, anticipate delayed consequences, and maintain stability across competing objectives under uncertainty.**

The AI Policy Engine fills this gap. It forces agents to:

- **Balance competing objectives** -- economy, environment, and society cannot all be maximised simultaneously
- **Reason about temporal dependencies** -- green investments take 3-6 steps to pay off; short-term thinking leads to collapse
- **Adapt to stochastic events** -- pandemics, recessions, and climate crises demand reactive policy shifts
- **Avoid catastrophic failures** -- crossing any of three collapse thresholds ends the episode immediately

This is not a toy benchmark. It is a compressed simulation of real governance, built to expose whether an AI agent can reason, plan, and adapt under the same constraints that make policy-making hard for humans.

---

## Who Is This For

This environment is designed for:

- **Researchers** studying multi-objective reinforcement learning and LLM-based decision-making
- **Developers** building autonomous agents that must reason about trade-offs and delayed consequences
- **Teams** evaluating AI systems for policy, governance, and resource allocation tasks
- **Safety researchers** analysing failure modes, collapse dynamics, and robustness in complex systems

It is particularly useful for benchmarking agents in scenarios where objectives conflict, consequences are delayed, and system stability is critical.

---

## Why This Problem Is Hard

- **No single policy optimises all objectives** -- every action improves one dimension at the cost of another
- **Effects are delayed and non-linear** -- investments take 3-6 steps to materialise; thresholds trigger exponential cascades
- **Events introduce stochastic disruptions** -- agents must adapt strategy mid-episode to survive pandemics, recessions, and crises
- **Collapse conditions require long-term planning** -- short-term greedy strategies inevitably trigger system failure

This makes the environment fundamentally different from traditional RL benchmarks where a single reward signal guides optimisation.

---

## How It Works

The simulation runs on a 21-dimensional state space representing a nation's environmental, economic, social, infrastructure, and policy metrics. Each step, the agent selects one of 16 policy levers. The world evolves through four layers:

```
Agent selects action (1 of 16 policy levers)
        |
        v
  Layer 1: Deterministic Effects
        |   Direct consequences. Tax increase -> GDP drops, revenue rises.
        v
  Layer 2: Non-linear Thresholds
        |   Tipping points activate. Pollution > 200 -> exponential health damage.
        v
  Layer 3: Delayed Effects Queue
        |   Past investments materialise. Education from step 5 boosts GDP at step 11.
        v
  Layer 4: Feedback Loops
        |   Systemic cascades. Low healthcare -> low productivity -> high unemployment -> low satisfaction.
        v
  Event Engine (seeded RNG)
        |   Stochastic shocks. Pandemic, trade war, tech breakthrough.
        v
  Explainability Engine
        |   Causal chain generated. "Pollution exceeded threshold -> healthcare degrading -> satisfaction at risk."
        v
  Reward Engine
        |   Multi-objective score: 30% economic + 30% environmental + 25% social + 15% stability - penalties.
        v
  Observation returned to agent
        Includes: 21 metrics, reward breakdown, causal explanation, risk alerts, event status.
```

---

## Example Episode

The following trace shows what happens when an agent prioritises short-term economic growth without balancing environmental and social consequences:

```
Step  1: expand_industry          GDP: 103 (+3)    Pollution: 128 (+8)   Satisfaction: 60
         Reasoning: GDP increased by 3.0 points; pollution increased by 8.0 (externality)

Step  5: expand_industry          GDP: 115 (+3)    Pollution: 168 (+8)   Satisfaction: 58
         ALERT: Pollution at 168, approaching danger zone
         Feedback: Pollution-Health loop now active (pollution > 150)

Step  8: expand_industry          GDP: 121 (+3)    Pollution: 204 (+8)   Satisfaction: 49
         CRITICAL: Pollution exceeded 200 (safe threshold)
         Effect: Healthcare and ecological stability now degrading exponentially

Step 12: no_action                GDP: 113         Pollution: 218        Satisfaction: 31
         Causal chain: [nonlinear] pollution > 200 -> exponential health damage
                       [feedback]  healthcare < 30 -> industrial output falling
                       [feedback]  inequality > 60 -> satisfaction eroding

Step 15: stimulate_economy        GDP: 108         Pollution: 238        Satisfaction: 18
         ALERT: Satisfaction at 18, approaching danger zone
         3 critical conditions active; satisfaction fell by 13.0

Step 20: COLLAPSED                GDP:  42         Pollution: 261        Satisfaction: 3
         Collapse trigger: public_satisfaction < 5
         Final score: 0.1203
         Causal chain: Unchecked pollution -> healthcare collapse -> productivity loss ->
                       GDP decline -> unemployment -> satisfaction spiral -> system failure
```

A smarter agent would have mixed in environmental controls (subsidize_renewables, enforce_emission_limits) and social investment (increase_welfare) to prevent the cascade. The environment rewards agents that maintain balance across all three dimensions, not agents that maximise any single metric.

---

## Common Failure Modes

Agents typically fail in one of three ways:

**1. Greedy Optimisation.** Maximising GDP through industrial expansion without controlling pollution leads to healthcare collapse, ecological degradation, and eventual system failure. The agent scores well on economic metrics for 10-15 steps before non-linear thresholds trigger an irreversible cascade.

**2. Overcorrection.** Aggressively pursuing environmental targets (repeated restrict_polluting_industries, implement_carbon_tax) collapses GDP and industrial output, triggering unemployment and satisfaction spirals. The agent achieves pollution targets but fails economically.

**3. Policy Instability.** Frequent switching between opposite actions (increase_tax then decrease_tax) triggers oscillation penalties and prevents delayed investments from materialising. The agent appears active but achieves nothing.

The environment is specifically designed to expose these weaknesses. Optimal performance requires a balanced, temporally-aware strategy that no single heuristic can achieve.

---

## Explainability Layer

Every observation includes a structured causal explanation:

```json
{
  "explanation": {
    "causal_chain": [
      {
        "layer": "deterministic",
        "trigger": "Action 'restrict_polluting_industries' executed",
        "effect": "Pollution reduced by 6.0 points",
        "severity": "info"
      },
      {
        "layer": "deterministic",
        "trigger": "Green regulation cost",
        "effect": "GDP decreased by 2.0 (trade-off)",
        "severity": "warning"
      },
      {
        "layer": "nonlinear",
        "trigger": "Pollution at 208 (above 200 threshold)",
        "effect": "Ongoing exponential health damage",
        "severity": "warning"
      },
      {
        "layer": "feedback",
        "trigger": "Inequality at 62 (above 60 threshold)",
        "effect": "Inequality-Satisfaction loop: public satisfaction being eroded",
        "severity": "warning"
      }
    ],
    "summary": "After 'restrict_polluting_industries': 2 warning(s); pollution_index fell by 6.0.",
    "risk_alerts": [
      "WARNING: Pollution at 208, approaching danger zone"
    ],
    "delta_report": {
      "pollution_index": -6.0,
      "gdp_index": -2.0,
      "carbon_emission_rate": -5.0,
      "unemployment_rate": 1.5
    }
  }
}
```

This provides:

- **Transparency** -- every state change has a traceable cause
- **Interpretability** -- agents and researchers can understand why metrics changed
- **Debugging** -- failed episodes can be diagnosed through the causal chain
- **Research value** -- enables study of how LLMs process and act on causal information

The explainability layer enables causal introspection of RL trajectories, making this environment suitable for safety-critical AI evaluation.

---

## Action Space

16 policy levers across 5 categories:

| Category | Actions | Primary Effect |
|----------|---------|----------------|
| Economic | `increase_tax`, `decrease_tax`, `stimulate_economy`, `reduce_interest_rates` | GDP, trade, investment |
| Industrial | `expand_industry`, `restrict_polluting_industries`, `incentivize_clean_tech` | Output, pollution, employment |
| Environmental | `enforce_emission_limits`, `subsidize_renewables`, `implement_carbon_tax` | Emissions, renewables, ecology |
| Social | `increase_welfare`, `invest_in_healthcare`, `invest_in_education` | Satisfaction, health, education |
| Infrastructure | `upgrade_energy_grid`, `invest_in_transport` | Efficiency (delayed) |

Every action has immediate effects, delayed effects (2-6 steps later), and hidden second-order consequences through feedback loops. There is no free lunch.

---

## Observation Space

21 continuous metrics plus temporal context, event status, reward breakdown, and causal explanation:

| Domain | Metrics | Range |
|--------|---------|-------|
| Environmental | `pollution_index`, `carbon_emission_rate`, `renewable_energy_ratio`, `ecological_stability` | 0-300, 0-100, 0-1, 0-100 |
| Economic | `gdp_index`, `industrial_output`, `unemployment_rate`, `inflation_rate`, `trade_balance`, `foreign_investment` | 0-200, 0-100, 0-50%, -10-30%, -100-100, 0-100 |
| Social | `public_satisfaction`, `healthcare_index`, `education_index`, `inequality_index` | 0-100 each |
| Infrastructure | `energy_efficiency`, `transport_efficiency` | 0-100 each |
| Policy | `tax_rate`, `regulation_strength`, `welfare_spending`, `green_subsidies`, `interest_rate` | 0-50%, 0-100, 0-100, 0-100, 0-20% |

---

## Non-linear Dynamics

The environment contains six threshold-based non-linear effects:

| Condition | Threshold | Consequence |
|-----------|-----------|-------------|
| Pollution catastrophe | pollution > 200 | Exponential healthcare and ecology damage |
| Tax overburden | tax_rate > 40% | Quadratic GDP suppression, investment flight |
| Mass unemployment | unemployment > 25% | Severe satisfaction drain, inequality surge |
| Hyperinflation | inflation > 15% | Satisfaction erosion, investment flight |
| GDP depression | gdp < 40 | Unemployment acceleration, satisfaction decay |
| Ecological tipping point | ecology < 20 | Runaway pollution increase |

---

## Feedback Loops

Six systemic loops create emergent behaviour:

| Loop | Condition | Effect |
|------|-----------|--------|
| Health-Productivity | healthcare < 30 | Industrial output falls, unemployment rises |
| Education-Innovation | education > 70 | GDP receives innovation bonus each step |
| Inequality-Unrest | inequality > 60 | Public satisfaction erodes |
| Unemployment-Social | unemployment > 18 | Satisfaction drops, inequality rises |
| Pollution-Health | pollution > 150 | Healthcare index degrades |
| Renewable Dividend | renewables > 30% | Automatic pollution and emission reduction |

---

## Event System

Eight stochastic event types with context-sensitive probabilities:

| Event | Base Rate | Duration | Impact |
|-------|-----------|----------|--------|
| Pandemic | 2.5%/step | 5 steps | GDP -3, satisfaction -4, unemployment +3 per step |
| Industrial Boom | 4.5%/step | 3 steps | GDP +3, pollution +5 per step |
| Climate Crisis | 3.5%/step | 4 steps | Pollution +8, ecology -5 per step |
| Public Protest | 5.5%/step | 2 steps | Satisfaction -5, regulation +3 per step |
| Tech Breakthrough | 3.5%/step | 3 steps | Renewables +0.03, efficiency +3 per step |
| Trade War | 3.0%/step | 4 steps | Trade balance -8, investment -5 per step |
| Natural Disaster | 2.5%/step | 2 steps | GDP -4, infrastructure -5, satisfaction -3 per step |
| Economic Recession | 2.5%/step | 5 steps | GDP -4, unemployment +3 per step |

Event probabilities adjust based on state: protests are more likely when satisfaction is low, climate crises trigger more frequently when pollution is high.

---

## Tasks and Grading

Four tasks with deterministic graders producing scores from 0.0 to 1.0:

### Task 1: Environmental Recovery (Easy, 50 steps)

**Objective**: Reduce pollution from 170 to below 80 while maintaining GDP above 60.

Events disabled. Tests basic understanding of green policy instruments and trade-off management. Starting state is designed to be recoverable — all agents (including Random) achieve 100% survival.

**Grader**: 50% pollution target achievement + 25% GDP preservation + 15% trajectory trend (pollution decreasing over time) + 10% ecological stability recovery.

### Task 2: Balanced Economy (Medium, 100 steps)

**Objective**: Simultaneously maintain GDP > 80, pollution < 100, and public satisfaction > 60.

Events at 50% frequency. Tests genuine multi-objective optimisation where no single policy achieves all three targets.

**Grader**: 50% fraction of steps meeting all 3 criteria + 30% final state composite score + 20% worst-metric floor (minimum of the 3 normalised metrics averaged over the trajectory).

### Task 3: Sustainable Governance (Hard, 200 steps)

**Objective**: Maintain long-horizon stability across all dimensions under calibrated event pressure.

Full event frequency with three calibration mechanisms:
- **Satisfaction event scaling (0.4x)** — event satisfaction impacts reduced to 40%
- **Floor damping (0.8)** — when satisfaction < 35, absorbs 80% of further losses (dual-phase: post-transition and post-events)
- **Crisis welfare bonus (+8.0)** — social actions grant bonus satisfaction when satisfaction < 40

These create a "recovery zone" that smart agents can exploit. Random agents achieve **2% survival** while crisis-responsive agents achieve **21% survival**.

**Grader**: 25% survival (no collapse) + 30% multi-metric balance across trajectory + 25% low volatility (penalises wild metric swings) + 20% event resilience (recovery speed after shocks).

### Task 4: Sustainable Governance — Extreme (200 steps)

**Objective**: Survive an unstable regime with irreversible cascade dynamics under full event pressure.

Full event frequency with **no satisfaction dampening (1.0x)**. This is a structural collapse regime — all tested strategies (Random, Heuristic, RL) collapse 100% of the time. Serves as a failure-mode analysis benchmark, not a learnable task.

**Grader**: Same as Task 3. The environment is designed to demonstrate that no reactive policy can overcome compounding stochastic satisfaction shocks.

> **Key finding**: Social stability acts as the primary failure bottleneck, dominating economic and environmental factors under stochastic pressure.

---

## Regime Analysis

We identify two distinct stability regimes for the governance task:

### Calibrated Regime (Task 3)

Under the full calibration mechanism (event scaling + floor damping + crisis bonus), survival becomes possible for agents that learn crisis-responsive behaviour. The **intelligence scaling curve** shows clear differentiation:

| Agent | Avg Score | Survival Rate | Avg Steps |
|-------|-----------|---------------|-----------|
| Greedy (GDP) | 0.2732 | 0% | 23 |
| Random | 0.2027 | 2% | 85 |
| Heuristic | 0.2280 | 11% | 110 |
| **Smart** | **0.2583** | **21%** | **127** |

### Extreme Regime (Task 4)

Under full event intensity with no recovery mechanisms, **all strategies collapse 100%** — this is a structural finding, not an agent limitation.

| Agent | Avg Score | Collapse Rate | Avg Steps |
|-------|-----------|---------------|-----------|
| Random | 0.2543 | 100% | 37 |
| Heuristic | 0.2559 | 100% | 62 |
| Smart | 0.2608 | 100% | 62 |

### Phase Transition Finding

> The phase transition is created by the floor damping + crisis bonus mechanism, not by event scaling alone. Even at 0.1x event scale without these mechanisms, the extreme regime maintains 0% survival. This is a *designed controllability boundary*, not an accidental threshold.

This dual-regime design allows the environment to function both as:
- **A failure-mode analysis tool** — the extreme regime reveals that satisfaction is the sole collapse vector
- **A learnable benchmark for policy agents** — the calibrated regime shows 10× survival improvement from Random to Smart

---

## Collapse Conditions

The episode terminates immediately if any threshold is breached:

| Condition | Threshold | Meaning |
|-----------|-----------|---------|
| GDP collapse | gdp_index < 15 | Total economic failure |
| Ecological collapse | pollution_index > 290 | Irreversible environmental destruction |
| Social collapse | public_satisfaction < 5 | Total loss of public trust |

---

## Reward Function

Dense, multi-objective, per-step reward in [-1, 1]:

```
reward = 0.30 * economic_score
       + 0.30 * environmental_score
       + 0.25 * social_score
       + 0.15 * stability_score
       - penalties
```

**Penalties** are applied for:
- Policy oscillation (ABAB action patterns)
- Opposite-action flip-flopping (increase_tax followed by decrease_tax)
- Collapse proximity (metric approaching threshold)
- Inaction under crisis (choosing no_action during a critical event)

---

## Setup

### Requirements

- Python 3.10+
- Docker (for containerised deployment)

### Installation

```bash
git clone <repo-url>
cd openenv
pip install -r requirements.txt
```

### Running the Server

```bash
# Direct
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# Docker
docker build -t ai-policy-engine .
docker run -p 7860:7860 ai-policy-engine
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment. Body: `{"seed": 42, "task_id": "environmental_recovery"}` |
| `/step` | POST | Execute action. Body: `{"action": {"action": "subsidize_renewables"}}` |
| `/state` | GET | Current state (episode_id, step_count) |
| `/schema` | GET | Action and observation JSON schemas |
| `/tasks` | GET | List all tasks with descriptions |

### Running Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-api-key"
python inference.py
```

The inference script combines a structured PolicyReasoner (rule-based pre-filter that identifies critical situations, generates reasoning, and ranks recommended actions) with LLM decision-making. It produces detailed per-step reasoning logs saved to `outputs/`.

---

## Baseline Scores

Produced with seed=42 using a simple heuristic strategy:

| Task | Steps | Score | Status |
|------|-------|-------|--------|
| Environmental Recovery | 50/50 | 0.89 | Completed |
| Balanced Economy | 100/100 | 0.23 | Completed |
| Sustainable Governance | 52/200 | 0.21 | Collapsed |

These scores represent a lower bound. A well-designed LLM agent with effective reasoning should significantly outperform these baselines on all three tasks.

---

## Reinforcement Learning Agent

The project includes a **curriculum-driven REINFORCE** (policy gradient) agent with reward shaping, implemented in pure Python (zero ML framework dependencies). The agent learns to govern by mapping the 21-dimensional normalised state to action probabilities through a 3-stage curriculum.

Architecture: 21 (state) → 96 (hidden, ReLU) → 16 (actions, softmax)

Features:
- **Curriculum learning** (Easy → Medium → Hard) with weight transfer between stages
- **Satisfaction-focused reward shaping** — exponential danger penalties when satisfaction approaches collapse threshold
- **Annealed entropy regularisation** (0.08 → 0.01) to explore first, exploit later
- **Cosine learning rate decay** with 5% warmup for stable convergence
- **High discount factor** (γ=0.997) for long-horizon planning on the hard task
- Gradient clipping, advantage normalisation, and best-weight checkpointing
- 5-phase action distribution tracking to visualise policy specialisation

### Training Results — Dual Regime (10,500 episodes, ~15 min)

| Task | Episodes | Best Train | RL Agent | Random | Heuristic | Survival |
|------|----------|------------|----------|--------|-----------|----------|
| Environmental Recovery | 3,000 | **0.9264** | **0.4263** | 0.3768 | 0.2706 | **48%** |
| Balanced Economy | 2,500 | **0.5647** | **0.1847** | 0.1802 | 0.2208 | 8% |
| Sustainable Governance | 5,000 | **0.4606** | 0.2345 | 0.2569 | 0.2229 | 0% |
| Sustainable Governance Extreme | eval only | 0.4341 | 0.2996 | 0.3371 | 0.2608 | 0% |

- **Easy task**: RL agent outperforms Random by **+13.1%** with **48% survival rate** (vs 32% for Random)
- **Calibrated hard task**: 68% longer episode survival (63 steps vs 37 in extreme). Best training score of **0.46** shows the agent finding viable trajectories during exploration
- **Extreme task**: All strategies collapse 100%. This is a structural finding, not an agent limitation

### Running the RL Agent

```bash
python rl_agent.py
```

This runs the full 3-stage curriculum, saves policy weights to `outputs/policy_*.json`, evaluates against Random and Heuristic baselines, then runs the Extreme regime evaluation. Full training report saved to `outputs/rl_training_report.json`.

---

## Validation Suite

Comprehensive 6-phase test suite (`python validation_suite.py`) — **all phases pass**:

| Phase | Test | Result |
|-------|------|--------|
| 1. Regime Validation | Difficulty hierarchy across 4 tasks | ✅ Easy 100% → Medium 18% → Hard 9.5% → Extreme 0% |
| 2. Intelligence Scaling | 4 agent types, survival differentiation | ✅ Random 2% → Heuristic 11% → Smart 21% |
| 3. Adversarial Robustness | Oscillation, spam, no-op, threshold exploit | ✅ Zero exploits found |
| 4. Causal Consistency | 688 explainability truth checks | ✅ 688/688 (100%) |
| 5. Determinism | Same seed ×3, different seed ×3 | ✅ Perfectly reproducible |
| 6. Phase Transition | Event scale sweep [0.1 → 1.0] | ✅ Calibration mechanism creates controllable boundary |

```bash
python validation_suite.py
```

---

## Project Structure

```
openenv/
    __init__.py                   Package exports
    models.py                     Typed Pydantic models (PolicyAction, RewardBreakdown)
    client.py                     EnvClient wrapper (WebSocket + HTTP fallback)
    inference.py                  Baseline script with PolicyReasoner + LLM
    rl_agent.py                   REINFORCE policy gradient RL agent
    generate_dashboard.py         Interactive dashboard generator
    dashboard.html                Self-contained visualization dashboard
    openenv.yaml                  OpenEnv manifest
    pyproject.toml                Package configuration
    requirements.txt              Dependencies
    Dockerfile                    Container (python:3.11-slim, non-root, port 7860)
    README.md                     This document
    server/
        __init__.py               Server package with clean exports
        app.py                    FastAPI application (dual-mode)
        policy_environment.py     Core Environment class (reset/step/state)
        transition_engine.py      4-layer transition system
        event_engine.py           Seeded random event engine (8 types)
        reward_engine.py          Multi-objective reward calculation
        explainability.py         Causal reasoning chain generator
        tasks.py                  Task definitions and deterministic graders
        config.py                 Constants, bounds, action definitions
    outputs/
        rl_training_report.json   RL learning curves and evaluation data
        policy_*.json             Trained policy weights per task
```

---

## OpenEnv Compliance

| Requirement | Status |
|-------------|--------|
| `reset()` returns initial observation | Yes |
| `step(action)` returns observation, reward, done | Yes |
| `state()` returns episode metadata | Yes |
| Typed Action model (Pydantic) | Yes |
| Typed Observation schema | Yes |
| `openenv.yaml` manifest | Yes |
| 3+ tasks with deterministic graders (0.0-1.0) | Yes |
| Dense reward signal | Yes |
| Reproducible with seed | Yes |
| Dockerfile builds and runs | Yes |
| HF Spaces deployment ready | Yes |
| Baseline inference script with reproducible scores | Yes |
| RL agent with learning curves | Yes |
| Interactive visualization dashboard | Yes |

---

## Visualization Dashboard

Generate the interactive dashboard:

```bash
python generate_dashboard.py
```

This produces a self-contained `dashboard.html` with 5 tabs:

| Tab | Contents |
|-----|----------|
| Overview | System architecture, collapse conditions, RL scores |
| Episode Explorer | Interactive metric charts, reward traces, step-by-step logs |
| Explainability | Causal chain viewer, risk alerts, per-step summaries |
| RL Learning Curves | Score/reward/collapse trends over training episodes |
| Policy Comparison | Bar charts comparing smart, heuristic, and greedy policies |

The dashboard uses Chart.js for rendering and requires no server -- open `dashboard.html` directly in any browser.

---

## License

MIT
