"""
ingestion/schemas.py

Pydantic models defining the shape of every record moving through
VoltSentinel's ingestion layer. Written first (Phase 2, file 1) because
db.py's table definitions, routes.py's request/response bodies, and
main.py all validate against these.

Field names/types are deliberately kept identical to what simulator/
actually produces (TelemetryGenerator, MaintenanceGenerator,
AttackInjector, DriverGenerator), so simulated CSVs can be loaded
straight into these models with zero renaming. This is also the
boundary where real BLE-connected BMS hardware would eventually plug
in — a real ingestion client would need to produce records matching
these same shapes.

Two fields — TelemetryReading.thermal_event_flag and
CommandEvent.is_attack — are simulator-only ground-truth labels used to
validate models/anomaly_detector.py during testing. They are Optional
and default to None specifically so real (non-simulated) ingestion
still validates cleanly without them. Detection/agent logic must never
read these fields as if they were legitimate signals.

Driver / VehicleAssignment (Feature 1 of the future roadmap — driver
identity & vehicle assignment) are additive: every existing signal in
the system stays keyed on vehicle_id only, and these two models don't
change any of that. They exist purely so a "who was driving this
vehicle, and when" dimension can be attached on top, matching
simulator/driver_generator.py's output shape exactly, the same way
MaintenanceTicket/CommandEvent already match maintenance_generator.py/
attack_injector.py.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ----------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------
class CommandType(str, Enum):
    DISCHARGE_CUTOFF = "discharge_cutoff"
    DISABLE = "disable"
    ENABLE = "enable"


# ----------------------------------------------------------------------
# Core record schemas — one per simulator output stream
# ----------------------------------------------------------------------
class TelemetryReading(BaseModel):
    """One row of battery telemetry. Matches TelemetryGenerator.generate_vehicle_telemetry()."""

    vehicle_id: str
    cycle: int = Field(..., ge=1)
    timestamp: datetime
    capacity_kwh: float = Field(..., gt=0)
    capacity_pct_of_rated: float = Field(..., ge=0, le=150)
    rated_capacity_kwh: float = Field(..., gt=0)
    voltage: float = Field(..., gt=0)
    temperature_c: float = Field(..., ge=-20, le=120)
    soc_pct: float = Field(..., ge=0, le=100)
    is_fast_charge: bool
    dod_pct: float = Field(..., ge=0, le=100)
    thermal_event_flag: Optional[bool] = Field(
        default=None,
        description="Simulator ground truth only — not a real detection signal.",
    )

    model_config = {"extra": "forbid"}

    @field_validator("vehicle_id")
    @classmethod
    def vehicle_id_format(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("vehicle_id must not be empty")
        return v


class MaintenanceTicket(BaseModel):
    """One maintenance ticket. Matches MaintenanceGenerator.generate_vehicle_tickets()."""

    ticket_id: str
    vehicle_id: str
    timestamp: datetime
    depot_lat: float = Field(..., ge=-90, le=90)
    depot_lon: float = Field(..., ge=-180, le=180)
    reason: str
    technician_id: str

    model_config = {"extra": "forbid"}


class CommandEvent(BaseModel):
    """One BMS control command — legitimate or unauthorized. Matches
    AttackInjector.generate_command_stream()."""

    command_id: str
    vehicle_id: str
    timestamp: datetime
    command_type: CommandType
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    ticket_id: Optional[str] = None
    is_attack: Optional[bool] = Field(
        default=None,
        description="Simulator ground truth only — the real detector must never read this.",
    )

    model_config = {"extra": "forbid"}


# ----------------------------------------------------------------------
# Driver identity / vehicle assignment (Future Roadmap Feature 1)
# ----------------------------------------------------------------------
class Driver(BaseModel):
    """One driver in the fleet's driver pool. Matches
    simulator.driver_generator.DriverGenerator.get_driver_pool()'s output.
    Independent of any specific vehicle — a driver exists in the pool
    before ever being assigned to one, mirroring how a real HR/roster
    system would onboard a driver before their first shift."""

    driver_id: str
    name: str
    license_id: str
    depot_home: str

    model_config = {"extra": "forbid"}

    @field_validator("driver_id")
    @classmethod
    def driver_id_format(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("driver_id must not be empty")
        return v


class VehicleMetadata(BaseModel):
    """Asset-registry record: make, model, VIN, purchase date, warranty expiry."""

    vehicle_id: str
    make: str
    model: str
    vin: str
    purchase_date: datetime
    warranty_expiry_date: datetime

    model_config = {"extra": "forbid"}

    @field_validator("vin")
    @classmethod
    def vin_format(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("vin must not be empty")
        return v


class VehicleAssignment(BaseModel):
    """One shift-assignment record: which driver had which vehicle over a
    given shift window. Matches
    simulator.driver_generator.DriverGenerator.generate_vehicle_assignments()'s
    output.

    shift_end is Optional because a shift can be open/ongoing — a live
    shift-handover system may not know the end time until the shift
    actually ends. The simulator itself always produces a concrete
    shift_end (it only ever generates *historical* assignment windows),
    but real ingestion should be able to accept an in-progress shift
    without a value here yet."""

    assignment_id: str
    vehicle_id: str
    driver_id: str
    shift_start: datetime
    shift_end: Optional[datetime] = None

    model_config = {"extra": "forbid"}


# ----------------------------------------------------------------------
# Batch request bodies — for bulk-loading simulator output over REST
# ----------------------------------------------------------------------
class TelemetryBatch(BaseModel):
    readings: List[TelemetryReading]


class MaintenanceBatch(BaseModel):
    tickets: List[MaintenanceTicket]


class CommandBatch(BaseModel):
    commands: List[CommandEvent]


class DriverBatch(BaseModel):
    drivers: List[Driver]


class VehicleAssignmentBatch(BaseModel):
    assignments: List[VehicleAssignment]


class VehicleMetadataBatch(BaseModel):
    vehicles: List[VehicleMetadata]


# ----------------------------------------------------------------------
# WebSocket envelope — for the live "Simulate Attack" streaming path
# ----------------------------------------------------------------------
class StreamMessageType(str, Enum):
    TELEMETRY = "telemetry"
    MAINTENANCE = "maintenance"
    COMMAND = "command"


class StreamMessage(BaseModel):
    """A single WebSocket frame. `payload` shape depends on `type` — routes.py
    dispatches on this field to pick which model to validate the payload against."""

    type: StreamMessageType
    payload: dict

    model_config = {"extra": "forbid"}


# ----------------------------------------------------------------------
# Ingestion response schemas
# ----------------------------------------------------------------------
class IngestionAck(BaseModel):
    status: str = "ok"
    records_received: int = 0
    records_inserted: int = 0
    errors: List[str] = Field(default_factory=list)