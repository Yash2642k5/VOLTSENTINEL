"""
agent/prompts.py

Reasoning prompt templates for the agent decision layer's LLM call
(agent/decision_engine.py, built next, calls Gemini per the build
order). This file only builds prompt text and defines the expected
response schema — it has no API client code and makes no network
calls, so it has zero dependency on which LLM provider is wired up
downstream.

Written before decision_engine.py and actions.py on purpose (per the
build order) so the JSON action schema is decided once here and both
of those files are built to match it, rather than the schema drifting
between prompt text and action-handling code.

Design choices, tied directly to the project doc:
  - The system prompt encodes the Perceive -> Reason -> Decide -> Act
    framing from §6.1 and the four action types + mocked functions
    from §6.2's table, so the model's output vocabulary matches
    agent/actions.py exactly.
  - The per-asset prompt passes the ALREADY-COMPUTED signals from
    models/risk_engine.py (RUL status, thermal counts, security
    severity, charging stress) rather than raw telemetry — the LLM's
    job is to weigh pre-scored signals together and decide priority/
    rationale, not to re-derive anomaly detection itself. Keeps the
    explainable, rule-based detection logic (§7) separate from the
    agent's reasoning step (§6), matching the architecture in §5.
  - A worked few-shot example is included so the model's rationale
    style stays consistent and auditable across assets, rather than
    varying with each call.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ----------------------------------------------------------------------
# Action vocabulary — must match agent/actions.py's mocked functions exactly
# ----------------------------------------------------------------------
ACTION_TYPES = (
    "maintenance_trigger",        # -> create_maintenance_ticket()
    "charge_policy_recommendation",  # -> recommend_charge_policy()
    "security_escalation",        # -> escalate_incident()
    "fleet_manager_notification", # -> notify_fleet_manager()
    "no_action",                  # explicit "nothing warranted" — never omit silently
)

PRIORITY_LEVELS = ("low", "medium", "high", "critical")


SYSTEM_PROMPT = """You are the decision layer of VoltSentinel, an AI Asset Performance \
Management agent for industrial EV battery fleets. You do not detect anomalies yourself \
— that is already done by upstream rule-based models. Your job is to REASON over a \
merged per-asset signal profile the way a human fleet asset manager would: weigh \
multiple signals together, decide what actions are warranted and at what priority, and \
explain your reasoning in terms of the specific signals observed.

You follow a strict Perceive -> Reason -> Decide -> Act loop:
  - Perceive: you are given RUL/degradation status, thermal-anomaly counts, security/\
command-anomaly severity, and charging-behaviour stress for one asset.
  - Reason: consider these signals TOGETHER, not independently. A moderate RUL decline \
combined with frequent fast-charging and a recent unticketed command is materially \
different from the same RUL decline in isolation — say so explicitly when it applies.
  - Decide: choose zero or more of the following action types, each with a priority:
      * maintenance_trigger — warranted when RUL status is degraded/critical, or thermal \
anomalies persist across multiple cycles.
      * charge_policy_recommendation — warranted when fast-charge frequency or depth-of-\
discharge is materially elevated versus the fleet baseline, especially if the stress \
trend is increasing.
      * security_escalation — warranted when there is an unticketed command AND (GPS \
mismatch OR a frequency spike) — i.e. security severity is medium or high. This is \
distinct from routine maintenance and must be flagged for fleet-manager review, not \
folded into a maintenance ticket.
      * fleet_manager_notification — warranted for any decision above priority "medium".
      * no_action — use this explicitly if, after reasoning over all signals together, \
nothing is currently warranted. Never simply omit output when there is nothing to do.
  - Act: your output IS the action list. You do not execute the actions yourself — a \
separate mocked function layer (agent/actions.py) does that from your output.

Every action MUST include a short, specific rationale tied to the actual signal values \
you were given (explainable and auditable) — never a generic statement like "asset needs \
attention." If you recommend nothing, say explicitly why the observed signals don't meet \
the bar for action.

Respond with ONLY a single JSON object matching this schema, no prose before or after:
{
  "vehicle_id": "<string, echoed from input>",
  "actions": [
    {
      "action_type": "<one of: maintenance_trigger | charge_policy_recommendation | \
security_escalation | fleet_manager_notification | no_action>",
      "priority": "<one of: low | medium | high | critical>",
      "rationale": "<specific, signal-grounded explanation, 1-3 sentences>",
      "parameters": { "<action-specific key-value pairs, e.g. reason, cap_charge_rate>" }
    }
  ],
  "summary": "<one-sentence overall assessment of this asset>"
}"""


# ----------------------------------------------------------------------
# Few-shot example — keeps rationale style consistent across the fleet
# ----------------------------------------------------------------------
FEW_SHOT_EXAMPLE_INPUT = """Asset: EVR-0042
RUL status: degraded (current capacity 76.2% of rated, projected RUL 42 cycles)
Thermal: 3 anomalies flagged in recent history, 0 critical-temperature readings
Security: severity=none, 0 unticketed commands, 0 suspicious commands
Charging: fast-charge frequency 28.0% (fleet baseline 24.5%), mean DoD 71.3%, stress trend stable"""

FEW_SHOT_EXAMPLE_OUTPUT = """{
  "vehicle_id": "EVR-0042",
  "actions": [
    {
      "action_type": "maintenance_trigger",
      "priority": "medium",
      "rationale": "RUL status is degraded at 76.2% capacity with only 42 cycles projected remaining, and 3 thermal anomalies have recurred recently — combined, this warrants a scheduled inspection before RUL crosses into critical range.",
      "parameters": {"reason": "degraded RUL with recurring thermal anomalies", "priority": "medium"}
    },
    {
      "action_type": "no_action",
      "priority": "low",
      "rationale": "Charging behaviour (28.0% fast-charge vs 24.5% fleet baseline, stable trend) is only mildly elevated and not worsening, so no charge-policy change is warranted yet.",
      "parameters": {}
    }
  ],
  "summary": "EVR-0042 needs a scheduled maintenance inspection soon due to degraded RUL and recurring thermal anomalies; charging behaviour is mildly elevated but stable and does not need intervention yet."
}"""


# ----------------------------------------------------------------------
# Per-asset prompt builder
# ----------------------------------------------------------------------
def _fmt(value: Any, suffix: str = "", none_text: str = "unknown") -> str:
    return f"{value}{suffix}" if value is not None else none_text


def format_asset_profile(profile: Dict[str, Any]) -> str:
    """Turns one row of models/risk_engine.py's merged profile (as a dict)
    into a readable text block for the prompt — natural-language framing
    reads more reliably for an LLM than a raw JSON dump of the same data."""
    return (
        f"Asset: {profile.get('vehicle_id', 'UNKNOWN')}\n"
        f"RUL status: {profile.get('status', 'unknown')} "
        f"(current capacity {_fmt(profile.get('current_capacity_pct'), '%')} of rated, "
        f"projected RUL {_fmt(profile.get('rul_cycles'), ' cycles')})\n"
        f"Thermal: {_fmt(profile.get('thermal_anomaly_count'), '', '0')} anomalies flagged "
        f"in recent history, {_fmt(profile.get('critical_temp_count'), '', '0')} "
        f"critical-temperature readings\n"
        f"Security: severity={profile.get('max_security_severity', 'none')}, "
        f"{_fmt(profile.get('unticketed_command_count'), '', '0')} unticketed commands, "
        f"{_fmt(profile.get('suspicious_command_count'), '', '0')} suspicious commands\n"
        f"Charging: fast-charge frequency {_fmt(profile.get('fast_charge_frequency_pct'), '%')} "
        f"(fleet baseline {_fmt(profile.get('fleet_fast_charge_baseline_pct'), '%')}), "
        f"mean DoD {_fmt(profile.get('mean_dod_pct'), '%')}, "
        f"stress trend {profile.get('stress_trend', 'unknown')}"
    )


def build_decision_prompt(profile: Dict[str, Any], include_few_shot: bool = True) -> str:
    """The full user-turn prompt for a single asset. decision_engine.py sends
    SYSTEM_PROMPT as the system message and this as the user message."""
    parts = []

    if include_few_shot:
        parts.append(
            "Here is one worked example of the expected reasoning style and output format:\n\n"
            f"Input:\n{FEW_SHOT_EXAMPLE_INPUT}\n\nOutput:\n{FEW_SHOT_EXAMPLE_OUTPUT}\n\n"
            "Now reason over the following asset in the same style."
        )

    parts.append(f"Input:\n{format_asset_profile(profile)}\n\nOutput:")
    return "\n\n".join(parts)


def build_batch_summary_prompt(profiles: list[Dict[str, Any]]) -> str:
    """Optional fleet-level summary prompt — not per-asset decisions (each
    asset should get its own build_decision_prompt call for auditability),
    but useful for a dashboard "fleet health summary" panel."""
    lines = [format_asset_profile(p) for p in profiles]
    joined = "\n\n".join(lines)
    return (
        f"Here are the current merged risk profiles for all {len(profiles)} assets in the fleet:\n\n"
        f"{joined}\n\n"
        "Write a concise 3-5 sentence fleet-level summary for a human fleet manager: "
        "call out the most urgent asset(s) by ID, any fleet-wide pattern across multiple "
        "assets (e.g. several vehicles trending toward high DoD), and one clear priority "
        "for today. Do not restate every asset individually — synthesize."
    )