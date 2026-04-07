"""
AI Policy Engine — Configuration & Constants

Defines the action space, state bounds, initial conditions for each task,
and all tunable parameters for the governance simulation.
"""

from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────
# Action Space
# ─────────────────────────────────────────────────────────────

VALID_ACTIONS: List[str] = [
    "no_action",
    "increase_tax",
    "decrease_tax",
    "stimulate_economy",
    "reduce_interest_rates",
    "expand_industry",
    "restrict_polluting_industries",
    "incentivize_clean_tech",
    "enforce_emission_limits",
    "subsidize_renewables",
    "implement_carbon_tax",
    "increase_welfare",
    "invest_in_healthcare",
    "invest_in_education",
    "upgrade_energy_grid",
    "invest_in_transport",
]

ACTION_DESCRIPTIONS: Dict[str, str] = {
    "no_action": "Take no policy action this turn.",
    "increase_tax": "Raise tax rates — boosts revenue but discourages investment.",
    "decrease_tax": "Lower tax rates — stimulates growth but reduces public services.",
    "stimulate_economy": "Inject stimulus — lowers unemployment but raises inflation and pollution.",
    "reduce_interest_rates": "Cut interest rates — cheaper borrowing, risk of inflation.",
    "expand_industry": "Expand industrial capacity — GDP and jobs up, pollution up.",
    "restrict_polluting_industries": "Restrict dirty industry — pollution down, jobs lost.",
    "incentivize_clean_tech": "Fund clean-tech R&D — delayed green benefits, upfront cost.",
    "enforce_emission_limits": "Impose strict emissions caps — fast pollution drop, industry hit.",
    "subsidize_renewables": "Subsidize renewable energy — large delayed green gains.",
    "implement_carbon_tax": "Tax carbon emissions — strong pollution reducer, hurts industry.",
    "increase_welfare": "Boost welfare spending — satisfaction up, fiscal cost.",
    "invest_in_healthcare": "Invest in healthcare — delayed health gains.",
    "invest_in_education": "Invest in education — long-term GDP and equality gains.",
    "upgrade_energy_grid": "Modernize energy infrastructure — delayed efficiency gains.",
    "invest_in_transport": "Improve transport networks — delayed efficiency and GDP.",
}

# ─────────────────────────────────────────────────────────────
# State Metric Bounds  (metric_name → (min, max))
# ─────────────────────────────────────────────────────────────

STATE_BOUNDS: Dict[str, Tuple[float, float]] = {
    # Environmental
    "pollution_index": (0.0, 300.0),
    "carbon_emission_rate": (0.0, 100.0),
    "renewable_energy_ratio": (0.0, 1.0),
    "ecological_stability": (0.0, 100.0),
    # Economic
    "gdp_index": (0.0, 200.0),
    "industrial_output": (0.0, 100.0),
    "unemployment_rate": (0.0, 50.0),
    "inflation_rate": (-10.0, 30.0),
    "trade_balance": (-100.0, 100.0),
    "foreign_investment": (0.0, 100.0),
    # Social
    "public_satisfaction": (0.0, 100.0),
    "healthcare_index": (0.0, 100.0),
    "education_index": (0.0, 100.0),
    "inequality_index": (0.0, 100.0),
    # Infrastructure
    "energy_efficiency": (0.0, 100.0),
    "transport_efficiency": (0.0, 100.0),
    # Policy knobs
    "tax_rate": (0.0, 50.0),
    "regulation_strength": (0.0, 100.0),
    "welfare_spending": (0.0, 100.0),
    "green_subsidies": (0.0, 100.0),
    "interest_rate": (0.0, 20.0),
}

# ─────────────────────────────────────────────────────────────
# Default (baseline) initial state
# ─────────────────────────────────────────────────────────────

DEFAULT_STATE: Dict[str, float] = {
    "pollution_index": 100.0,
    "carbon_emission_rate": 50.0,
    "renewable_energy_ratio": 0.20,
    "ecological_stability": 70.0,
    "gdp_index": 100.0,
    "industrial_output": 60.0,
    "unemployment_rate": 8.0,
    "inflation_rate": 3.0,
    "trade_balance": 5.0,
    "foreign_investment": 50.0,
    "public_satisfaction": 60.0,
    "healthcare_index": 55.0,
    "education_index": 50.0,
    "inequality_index": 40.0,
    "energy_efficiency": 50.0,
    "transport_efficiency": 50.0,
    "tax_rate": 25.0,
    "regulation_strength": 40.0,
    "welfare_spending": 30.0,
    "green_subsidies": 10.0,
    "interest_rate": 5.0,
}

# ─────────────────────────────────────────────────────────────
# Collapse thresholds — episode ends with failure
# ─────────────────────────────────────────────────────────────

COLLAPSE_CONDITIONS: Dict[str, Tuple[str, float]] = {
    "gdp_collapse": ("gdp_index", 15.0),            # GDP drops below 15
    "eco_collapse": ("pollution_index", 290.0),      # Pollution exceeds 290
    "social_collapse": ("public_satisfaction", 5.0), # Satisfaction below 5
}

# ─────────────────────────────────────────────────────────────
# Reward weights
# ─────────────────────────────────────────────────────────────

REWARD_WEIGHTS = {
    "economic": 0.30,
    "environmental": 0.30,
    "social": 0.25,
    "stability": 0.15,
}

# Key metrics tracked for stability / volatility scoring
KEY_METRICS: List[str] = [
    "pollution_index", "gdp_index", "public_satisfaction",
    "healthcare_index", "unemployment_rate", "renewable_energy_ratio",
]

# ─────────────────────────────────────────────────────────────
# Task-specific initial states & parameters
# ─────────────────────────────────────────────────────────────

TASK_CONFIGS: Dict[str, dict] = {
    "environmental_recovery": {
        "description": (
            "Environmental Recovery (Easy): Reduce dangerously high pollution to "
            "safe levels (below 80) while keeping GDP above 60. "
            "Events are disabled — focus purely on green policy."
        ),
        "max_steps": 50,
        "events_enabled": False,
        "event_frequency_multiplier": 0.0,
        "initial_state_overrides": {
            "pollution_index": 170.0,       # high but below exponential threshold (180)
            "carbon_emission_rate": 65.0,   # elevated but manageable
            "renewable_energy_ratio": 0.15,
            "ecological_stability": 55.0,   # stressed but recoverable
            "gdp_index": 105.0,
            "industrial_output": 65.0,
            "unemployment_rate": 5.0,
            "public_satisfaction": 70.0,    # healthy start
            "healthcare_index": 55.0,
            "inequality_index": 30.0,
        },
    },
    "balanced_economy": {
        "description": (
            "Balanced Economy (Medium): Simultaneously maintain GDP > 80, "
            "pollution < 100, and public satisfaction > 60 over 100 steps. "
            "Random events occur at reduced frequency."
        ),
        "max_steps": 100,
        "events_enabled": True,
        "event_frequency_multiplier": 0.5,
        "initial_state_overrides": {
            "pollution_index": 140.0,
            "carbon_emission_rate": 60.0,
            "renewable_energy_ratio": 0.15,
            "ecological_stability": 55.0,
            "gdp_index": 85.0,
            "industrial_output": 55.0,
            "unemployment_rate": 12.0,
            "inflation_rate": 5.0,
            "public_satisfaction": 48.0,
            "healthcare_index": 48.0,
            "education_index": 45.0,
            "inequality_index": 48.0,
        },
    },
    "sustainable_governance": {
        "description": (
            "Sustainable Governance (Hard): Maintain stability across all dimensions "
            "for 200 steps under stochastic events with calibrated intensity. "
            "Satisfaction shocks are dampened, making survival possible but non-trivial. "
            "Agents must actively manage social stability to avoid collapse."
        ),
        "max_steps": 200,
        "events_enabled": True,
        "event_frequency_multiplier": 1.0,
        "satisfaction_event_scale": 0.4,    # calibrated: 40% event satisfaction impact
        "satisfaction_floor_damping": 0.8,  # when sat < 35, absorb 80% of losses
        "crisis_welfare_bonus": 8.0,        # extra +8.0 sat when social actions used during crisis
        "initial_state_overrides": {
            "pollution_index": 130.0,
            "carbon_emission_rate": 55.0,
            "renewable_energy_ratio": 0.18,
            "ecological_stability": 60.0,
            "gdp_index": 90.0,
            "industrial_output": 58.0,
            "unemployment_rate": 7.0,
            "inflation_rate": 4.0,
            "trade_balance": 0.0,
            "foreign_investment": 45.0,
            "public_satisfaction": 65.0,    # higher starting buffer for calibrated regime
            "healthcare_index": 50.0,
            "education_index": 48.0,
            "inequality_index": 38.0,
            "energy_efficiency": 45.0,
            "transport_efficiency": 45.0,
            "tax_rate": 28.0,
            "regulation_strength": 35.0,
            "welfare_spending": 25.0,
            "green_subsidies": 8.0,
            "interest_rate": 6.0,
        },
    },
    "sustainable_governance_extreme": {
        "description": (
            "Sustainable Governance — EXTREME: Unstable regime with irreversible "
            "cascade dynamics under full event pressure. All tested strategies "
            "(Random, Heuristic, RL) collapse 100% of the time. Satisfaction shocks "
            "dominate, revealing a structural instability that no reactive policy "
            "can overcome. Serves as a failure-mode analysis benchmark."
        ),
        "max_steps": 200,
        "events_enabled": True,
        "event_frequency_multiplier": 1.0,
        "satisfaction_event_scale": 1.0,    # full intensity: structural collapse
        "initial_state_overrides": {
            "pollution_index": 130.0,
            "carbon_emission_rate": 55.0,
            "renewable_energy_ratio": 0.18,
            "ecological_stability": 60.0,
            "gdp_index": 90.0,
            "industrial_output": 58.0,
            "unemployment_rate": 10.0,
            "inflation_rate": 4.0,
            "trade_balance": 0.0,
            "foreign_investment": 45.0,
            "public_satisfaction": 52.0,    # original low starting satisfaction
            "healthcare_index": 50.0,
            "education_index": 48.0,
            "inequality_index": 45.0,
            "energy_efficiency": 45.0,
            "transport_efficiency": 45.0,
            "tax_rate": 28.0,
            "regulation_strength": 35.0,
            "welfare_spending": 25.0,
            "green_subsidies": 8.0,
            "interest_rate": 6.0,
        },
    },
}
