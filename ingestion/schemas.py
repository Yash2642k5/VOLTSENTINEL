from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# Enums
class CommandType(str, Enum):
    DISCHARGE_CUTOFF = "discharge_cutoff"
    DISABLE = "disable"
    ENABLE = "enable"

# Core record schemas — one per simulator output stream
class TelemetryReading(BaseModel):

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

    ticket_id: str
    vehicle_id: str
    timestamp: datetime
    depot_lat: float = Field(..., ge=-90, le=90)
    depot_lon: float = Field(..., ge=-180, le=180)
    reason: str
    technician_id: str

    model_config = {"extra": "forbid"}


class CommandEvent(BaseModel):

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

class Driver(BaseModel):

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

    assignment_id: str
    vehicle_id: str
    driver_id: str
    shift_start: datetime
    shift_end: Optional[datetime] = None

    model_config = {"extra": "forbid"}

# Batch request bodies

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

# WebSocket envelope — for the live "Simulate Attack" streaming path

class StreamMessageType(str, Enum):
    TELEMETRY = "telemetry"
    MAINTENANCE = "maintenance"
    COMMAND = "command"


class StreamMessage(BaseModel):

    type: StreamMessageType
    payload: dict

    model_config = {"extra": "forbid"}

# Ingestion response schemas
class IngestionAck(BaseModel):
    status: str = "ok"
    records_received: int = 0
    records_inserted: int = 0
    errors: List[str] = Field(default_factory=list)