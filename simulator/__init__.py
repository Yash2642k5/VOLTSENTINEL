"""
VoltSentinel — Simulator Package

Generates synthetic EV battery telemetry, mock maintenance tickets, and
injects Tirri Challenge-style unauthorized BMS command events for a
simulated fleet. This package has no dependency on any other VoltSentinel
component (ingestion, models, agent, dashboard) — it is pure data generation.
"""

from simulator.config import SimulatorConfig

__all__ = ["SimulatorConfig"]