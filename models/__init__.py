"""
VoltSentinel — Models Package

Parallel analytical layers reading from SQLite (ingestion/db.py):
RUL regression, command/thermal anomaly detection, and charging-pattern
analysis. Each is independent of the others; risk_engine.py merges
their outputs into one per-asset profile for the agent layer.
"""