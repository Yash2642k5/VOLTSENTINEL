"""
simulator/config.py

Single source of truth for every parameter the simulator uses:
fleet size, cycle count, battery decay behaviour, thermal thresholds,
charging-pattern behaviour, maintenance ticket generation, and
Tirri Challenge-style attack injection.

Everything else in simulator/ (telemetry_generator, maintenance_generator,
attack_injector) reads from SimulatorConfig — no magic numbers should
live anywhere else in the simulator package.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SimulatorConfig:
    # ------------------------------------------------------------------
    # Fleet / run parameters
    # ------------------------------------------------------------------
    fleet_size: int = 50                 # number of vehicles simulated
    num_cycles: int = 500                # charge-discharge cycles per vehicle
    random_seed: int = 42                # reproducibility across runs

    sim_start_date: str = "2026-01-01"   # ISO date; telemetry timestamps start here
    avg_cycles_per_day: float = 1.5      # controls how timestamps are spaced out

    # ------------------------------------------------------------------
    # Battery identity / capacity
    # ------------------------------------------------------------------
    rated_capacity_kwh: float = 3.5      # nominal pack capacity (e-rickshaw scale)
    capacity_variance_pct: float = 0.05  # per-vehicle manufacturing variance

    # ------------------------------------------------------------------
    # Degradation model (exponential capacity fade + noise)
    # capacity(cycle) = rated_capacity * exp(-decay_rate * cycle) + noise
    # ------------------------------------------------------------------
    decay_rate_mean: float = 0.0009      # mean per-cycle decay rate
    decay_rate_std: float = 0.0002       # per-vehicle variance in decay rate
    capacity_noise_std_pct: float = 0.01 # gaussian noise added per reading
    end_of_life_capacity_pct: float = 0.70  # RUL "dead" threshold (70% of rated)

    # ------------------------------------------------------------------
    # Voltage / SoC behaviour
    # ------------------------------------------------------------------
    nominal_voltage: float = 51.2        # nominal pack voltage (48V-class LFP)
    voltage_noise_std: float = 0.3
    soc_min_pct: float = 5.0             # simulated vehicles rarely fully deplete
    soc_max_pct: float = 100.0

    # ------------------------------------------------------------------
    # Thermal behaviour
    # ------------------------------------------------------------------
    ambient_temp_mean_c: float = 32.0    # India-representative ambient baseline
    ambient_temp_std_c: float = 4.0
    charging_temp_rise_c: float = 8.0    # typical rise while fast-charging
    safe_temp_threshold_c: float = 45.0  # sustained-above-this = thermal flag
    critical_temp_threshold_c: float = 55.0
    thermal_event_probability: float = 0.03   # chance any given cycle has a thermal excursion
    thermal_ramp_abnormal_c_per_min: float = 2.0  # abnormal rate-of-change threshold

    # ------------------------------------------------------------------
    # Charging-pattern behaviour
    # ------------------------------------------------------------------
    fast_charge_probability: float = 0.30      # fraction of charge events that are fast-charge
    fast_charge_stress_multiplier: float = 1.8 # extra decay stress applied on fast-charge cycles
    dod_mean_pct: float = 65.0                 # mean depth-of-discharge per cycle
    dod_std_pct: float = 15.0
    high_dod_threshold_pct: float = 85.0       # DoD above this counts as "abusive"
    fleet_fast_charge_baseline_pct: float = 25.0  # fleet-wide baseline used by charging_analyzer

    # ------------------------------------------------------------------
    # Demo guarantee — force one vehicle to show charging-stress behaviour
    # so the "suggested charge policy" UI path is always demonstrable,
    # instead of depending on random chance across the fleet.
    # ------------------------------------------------------------------
    demo_stressed_vehicle_index: Optional[int] = 0   # index into vehicle_ids, e.g. EVR-0001
    demo_stressed_fast_charge_probability: float = 0.85
    demo_stressed_dod_mean_pct: float = 92.0

    # ------------------------------------------------------------------
    # Maintenance ticket generation
    # ------------------------------------------------------------------
    maintenance_events_per_vehicle: Tuple[int, int] = (2, 6)  # (min, max) legit tickets per vehicle
    depot_locations: Tuple[Tuple[float, float], ...] = (
        (12.9716, 77.5946),   # Bengaluru depot
        (28.7041, 77.1025),   # Delhi depot
        (19.0760, 72.8777),   # Mumbai depot
    )
    depot_gps_jitter_deg: float = 0.003   # ~300m jitter around a depot for realism
    maintenance_reasons: Tuple[str, ...] = (
        "Scheduled inspection",
        "Battery diagnostics",
        "Firmware update",
        "Discharge cut-off test",
        "Thermal sensor check",
        "Post-incident inspection",
    )

    # ------------------------------------------------------------------
    # Attack injection (Tirri Challenge-style unauthorized commands)
    # ------------------------------------------------------------------
    attack_injection_rate_pct: float = 0.08   # fraction of vehicles that get >=1 attack event
    attacks_per_affected_vehicle: Tuple[int, int] = (1, 3)  # (min, max) attack events

    # individual attack signal toggles/weights — used to vary which
    # suspicious signal(s) a given injected attack exhibits
    attack_no_ticket_weight: float = 1.0       # always true for injected attacks
    attack_gps_mismatch_probability: float = 0.7   # vehicle shown away from any depot
    attack_road_gps_jitter_deg: float = 0.05       # spread of "in traffic" coordinates

    attack_frequency_burst_probability: float = 0.5   # some attacks come as a repeated burst
    attack_burst_command_count: Tuple[int, int] = (3, 8)  # commands fired within the burst window
    attack_burst_window_seconds: int = 120                # burst window length

    command_types: Tuple[str, ...] = ("discharge_cutoff", "disable", "enable")
    attack_command_types: Tuple[str, ...] = ("discharge_cutoff", "disable")

    # ------------------------------------------------------------------
    # Output paths (relative to project root)
    # ------------------------------------------------------------------
    seed_dir: str = "data/seed"
    seed_telemetry_filename: str = "sample_telemetry.csv"
    seed_attacks_filename: str = "sample_attacks.csv"


# Default shared instance — import this directly for standalone scripts
# and quick testing (`from simulator.config import default_config`).
default_config = SimulatorConfig()