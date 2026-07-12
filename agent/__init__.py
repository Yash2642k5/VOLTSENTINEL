"""
VoltSentinel — Agent Package

The Perceive -> Reason -> Decide -> Act loop (project doc §6) that sits
above models/risk_engine.py. Reads the merged per-asset risk profile,
reasons over it with an LLM call, and emits concrete mocked actions
(maintenance triggers, charge-policy recommendations, security
escalations, fleet-manager notifications).
"""