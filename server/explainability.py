"""
AI Policy Engine -- Explainability Layer

Generates human-readable causal explanations for why the world
state changed between steps. Produces a structured reasoning chain
showing which transition layers fired, what thresholds were crossed,
and what feedback loops activated.

This is the environment's "glass box" -- it makes the simulation
transparent and interpretable, which is critical for:
  - Debugging agent behaviour
  - Understanding policy trade-offs
  - Research into interpretable RL
  - Benchmarking LLM reasoning about causal chains
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import COLLAPSE_CONDITIONS, STATE_BOUNDS


# ------------------------------------------------------------------
# Causal chain types
# ------------------------------------------------------------------

class CausalLink:
    """One link in the causal reasoning chain."""

    __slots__ = ("layer", "trigger", "effect", "severity")

    def __init__(self, layer: str, trigger: str, effect: str, severity: str = "info"):
        self.layer = layer        # deterministic | nonlinear | delayed | feedback | event
        self.trigger = trigger    # what caused it
        self.effect = effect      # what happened
        self.severity = severity  # info | warning | critical

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "trigger": self.trigger,
            "effect": self.effect,
            "severity": self.severity,
        }

    def __repr__(self) -> str:
        tag = {"info": " ", "warning": "!", "critical": "X"}[self.severity]
        return f"[{tag}] [{self.layer}] {self.trigger} -> {self.effect}"


# ------------------------------------------------------------------
# Explainability engine
# ------------------------------------------------------------------

class ExplainabilityEngine:
    """
    Analyses pre/post state diffs and action context to produce
    a structured causal explanation for each step.
    """

    def explain(
        self,
        action: str,
        prev_state: Optional[Dict[str, float]],
        curr_state: Dict[str, float],
        active_events: List[str],
        step: int,
    ) -> Dict:
        """
        Generate a full explanation for one step.

        Returns a dict with:
          - causal_chain: list of CausalLink dicts
          - summary: one-sentence human-readable summary
          - risk_alerts: list of approaching-threshold warnings
          - delta_report: top metric changes
        """
        if prev_state is None:
            return {
                "causal_chain": [],
                "summary": "Episode initialised.",
                "risk_alerts": [],
                "delta_report": {},
            }

        chain: List[CausalLink] = []
        deltas = self._compute_deltas(prev_state, curr_state)

        # -- Layer 1: Deterministic action effects --
        chain.extend(self._explain_action(action, deltas))

        # -- Layer 2: Non-linear threshold effects --
        chain.extend(self._explain_nonlinear(prev_state, curr_state))

        # -- Layer 4: Feedback loops --
        chain.extend(self._explain_feedback(curr_state))

        # -- Events --
        chain.extend(self._explain_events(active_events))

        # -- Risk alerts (approaching collapse) --
        risk_alerts = self._check_risk_proximity(curr_state)

        # -- Build summary --
        summary = self._build_summary(action, deltas, chain, risk_alerts)

        # -- Top deltas --
        sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
        delta_report = {k: round(v, 2) for k, v in sorted_deltas[:8] if abs(v) > 0.5}

        return {
            "causal_chain": [c.to_dict() for c in chain],
            "summary": summary,
            "risk_alerts": risk_alerts,
            "delta_report": delta_report,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_deltas(prev: Dict[str, float], curr: Dict[str, float]) -> Dict[str, float]:
        deltas = {}
        for k in curr:
            if k in prev:
                d = curr[k] - prev[k]
                if abs(d) > 0.01:
                    deltas[k] = d
        return deltas

    @staticmethod
    def _explain_action(action: str, deltas: Dict[str, float]) -> List[CausalLink]:
        chain = []
        if action == "no_action":
            return chain

        # Map action categories
        green_actions = {
            "restrict_polluting_industries", "enforce_emission_limits",
            "subsidize_renewables", "implement_carbon_tax", "incentivize_clean_tech",
        }
        econ_actions = {
            "increase_tax", "decrease_tax", "stimulate_economy",
            "reduce_interest_rates", "expand_industry",
        }
        social_actions = {"increase_welfare", "invest_in_healthcare", "invest_in_education"}
        infra_actions = {"upgrade_energy_grid", "invest_in_transport"}

        if action in green_actions:
            if "pollution_index" in deltas and deltas["pollution_index"] < 0:
                chain.append(CausalLink(
                    "deterministic", f"Action '{action}' executed",
                    f"Pollution reduced by {abs(deltas['pollution_index']):.1f} points",
                ))
            if "gdp_index" in deltas and deltas["gdp_index"] < 0:
                chain.append(CausalLink(
                    "deterministic", f"Green regulation cost",
                    f"GDP decreased by {abs(deltas['gdp_index']):.1f} (trade-off)",
                    "warning",
                ))
        elif action in econ_actions:
            if "gdp_index" in deltas and deltas["gdp_index"] > 0:
                chain.append(CausalLink(
                    "deterministic", f"Action '{action}' executed",
                    f"GDP increased by {deltas['gdp_index']:.1f} points",
                ))
            if "pollution_index" in deltas and deltas["pollution_index"] > 0:
                chain.append(CausalLink(
                    "deterministic", f"Economic expansion side-effect",
                    f"Pollution increased by {deltas['pollution_index']:.1f} (externality)",
                    "warning",
                ))
        elif action in social_actions:
            chain.append(CausalLink(
                "deterministic", f"Action '{action}' executed",
                "Social investment initiated (effects may be delayed)",
            ))
        elif action in infra_actions:
            chain.append(CausalLink(
                "deterministic", f"Action '{action}' executed",
                "Infrastructure investment initiated (delayed returns expected)",
            ))

        return chain

    @staticmethod
    def _explain_nonlinear(prev: Dict[str, float], curr: Dict[str, float]) -> List[CausalLink]:
        chain = []

        # Pollution catastrophe threshold
        if prev.get("pollution_index", 0) <= 200 < curr.get("pollution_index", 0):
            chain.append(CausalLink(
                "nonlinear",
                "Pollution exceeded 200 (safe threshold)",
                "Healthcare and ecological stability now degrading exponentially",
                "critical",
            ))
        elif curr.get("pollution_index", 0) > 200:
            chain.append(CausalLink(
                "nonlinear",
                f"Pollution at {curr['pollution_index']:.0f} (above 200 threshold)",
                "Ongoing exponential health damage",
                "warning",
            ))

        # Tax overburden
        if curr.get("tax_rate", 0) > 40:
            chain.append(CausalLink(
                "nonlinear",
                f"Tax rate at {curr['tax_rate']:.0f}% (above 40% threshold)",
                "Quadratic GDP suppression active",
                "warning",
            ))

        # GDP depression
        if curr.get("gdp_index", 100) < 40:
            chain.append(CausalLink(
                "nonlinear",
                f"GDP at {curr['gdp_index']:.0f} (below 40 depression threshold)",
                "Unemployment rising, satisfaction dropping from economic depression",
                "critical",
            ))

        # Mass unemployment
        if curr.get("unemployment_rate", 0) > 25:
            chain.append(CausalLink(
                "nonlinear",
                f"Unemployment at {curr['unemployment_rate']:.0f}% (crisis level)",
                "Severe satisfaction drain and rising inequality",
                "critical",
            ))

        return chain

    @staticmethod
    def _explain_feedback(curr: Dict[str, float]) -> List[CausalLink]:
        chain = []

        if curr.get("healthcare_index", 50) < 30:
            chain.append(CausalLink(
                "feedback",
                f"Healthcare index at {curr['healthcare_index']:.0f} (critically low)",
                "Health-Productivity loop: industrial output reduced, unemployment rising",
                "warning",
            ))

        if curr.get("education_index", 50) > 70:
            chain.append(CausalLink(
                "feedback",
                f"Education index at {curr['education_index']:.0f} (high)",
                "Education-Innovation loop: GDP receiving innovation bonus each step",
            ))

        if curr.get("inequality_index", 40) > 60:
            chain.append(CausalLink(
                "feedback",
                f"Inequality at {curr['inequality_index']:.0f} (above 60 threshold)",
                "Inequality-Satisfaction loop: public satisfaction being eroded",
                "warning",
            ))

        rer = curr.get("renewable_energy_ratio", 0)
        if rer > 0.3:
            chain.append(CausalLink(
                "feedback",
                f"Renewable ratio at {rer:.0%} (above 30% threshold)",
                "Renewable dividend: automatic pollution and emission reduction",
            ))

        return chain

    @staticmethod
    def _explain_events(active_events: List[str]) -> List[CausalLink]:
        chain = []
        event_impacts = {
            "pandemic": "GDP falling, unemployment rising, healthcare strained",
            "industrial_boom": "GDP and industry surging but pollution increasing",
            "climate_crisis": "Pollution spiking, ecology destabilised",
            "public_protest": "Satisfaction dropping, foreign investment fleeing",
            "tech_breakthrough": "Renewables and efficiency accelerating",
            "trade_war": "Trade balance collapsing, investment declining",
            "natural_disaster": "Infrastructure damaged, GDP hit, public distress",
            "economic_recession": "Broad economic contraction, unemployment rising",
        }
        for event in active_events:
            impact = event_impacts.get(event, "Unknown effects")
            chain.append(CausalLink(
                "event",
                f"Event active: {event.replace('_', ' ').upper()}",
                impact,
                "critical" if event in ("pandemic", "natural_disaster", "economic_recession") else "warning",
            ))
        return chain

    @staticmethod
    def _check_risk_proximity(curr: Dict[str, float]) -> List[str]:
        """Return warnings for metrics approaching collapse thresholds."""
        alerts = []

        gdp = curr.get("gdp_index", 100)
        if gdp < 35:
            alerts.append(f"CRITICAL: GDP at {gdp:.0f}, collapse threshold is 15")
        elif gdp < 50:
            alerts.append(f"WARNING: GDP at {gdp:.0f}, approaching danger zone")

        poll = curr.get("pollution_index", 100)
        if poll > 260:
            alerts.append(f"CRITICAL: Pollution at {poll:.0f}, collapse threshold is 290")
        elif poll > 220:
            alerts.append(f"WARNING: Pollution at {poll:.0f}, approaching danger zone")

        sat = curr.get("public_satisfaction", 50)
        if sat < 15:
            alerts.append(f"CRITICAL: Satisfaction at {sat:.0f}, collapse threshold is 5")
        elif sat < 25:
            alerts.append(f"WARNING: Satisfaction at {sat:.0f}, approaching danger zone")

        return alerts

    @staticmethod
    def _build_summary(
        action: str,
        deltas: Dict[str, float],
        chain: List[CausalLink],
        risk_alerts: List[str],
    ) -> str:
        """Build a one-sentence summary of this step."""
        parts = []

        # Count severity
        crits = sum(1 for c in chain if c.severity == "critical")
        warns = sum(1 for c in chain if c.severity == "warning")

        if crits > 0:
            parts.append(f"{crits} critical condition(s) active")
        if warns > 0:
            parts.append(f"{warns} warning(s)")

        # Top change
        if deltas:
            top_k = sorted(deltas.keys(), key=lambda k: abs(deltas[k]), reverse=True)[0]
            direction = "rose" if deltas[top_k] > 0 else "fell"
            parts.append(f"{top_k} {direction} by {abs(deltas[top_k]):.1f}")

        if risk_alerts:
            parts.append(f"{len(risk_alerts)} risk alert(s)")

        if not parts:
            return f"Action '{action}' applied with minimal state change."

        return f"After '{action}': " + "; ".join(parts) + "."
