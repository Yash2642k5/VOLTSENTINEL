"""
ingestion/db.py

SQLite connection helpers and table creation, matching the shapes
defined in schemas.py exactly. routes.py writes simulator/live data in
through the insert_* functions here; models/ (Phase 3) reads it back
out through the get_* functions.

Tables:
    telemetry            <- TelemetryReading
    maintenance_tickets  <- MaintenanceTicket
    commands             <- CommandEvent  (ticket_id FK -> maintenance_tickets)
    vehicle_quarantine   <- circuit-breaker state per vehicle (Tier 3 agent action)
    rejected_commands    <- audit trail of commands rejected due to quarantine

Inserts use INSERT OR IGNORE against natural unique keys, so re-running
ingestion on the same simulator output (e.g. during dev/demo rehearsal)
is idempotent instead of raising or duplicating rows.

Quarantine enforcement: if a vehicle is quarantined (vehicle_quarantine.active
= 1), insert_command / insert_command_batch reject any INCOMING command for
that vehicle that has no matching maintenance ticket — real enforcement at
the ingestion boundary, not just a detection flag applied after the fact.
A ticketed command (a real technician visit) always still goes through,
quarantine or not.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from .schemas import CommandEvent, MaintenanceTicket, TelemetryReading

DB_PATH = os.path.join("data", "voltsentinel.db")


# ----------------------------------------------------------------------
# Connection
# ----------------------------------------------------------------------
def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    # check_same_thread=False: dashboard/utils.py wraps this connection in
    # st.cache_resource, so the SAME connection object is reused across every
    # Streamlit rerun -- including the rerun triggered by a widget callback
    # (e.g. the "Simulate Attack" button), which Streamlit executes on a
    # different thread than the one that first created this connection.
    # sqlite3 refuses cross-thread use by default. Safe to disable that check
    # here specifically because Streamlit processes reruns for a given
    # session sequentially, never concurrently, so there is no risk of two
    # threads touching the connection at the same instant.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ----------------------------------------------------------------------
# Schema creation
# ----------------------------------------------------------------------
_CREATE_TELEMETRY = """
CREATE TABLE IF NOT EXISTS telemetry (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id              TEXT NOT NULL,
    cycle                   INTEGER NOT NULL,
    timestamp               TEXT NOT NULL,
    capacity_kwh            REAL NOT NULL,
    capacity_pct_of_rated   REAL NOT NULL,
    rated_capacity_kwh      REAL NOT NULL,
    voltage                 REAL NOT NULL,
    temperature_c           REAL NOT NULL,
    soc_pct                 REAL NOT NULL,
    is_fast_charge          INTEGER NOT NULL,
    dod_pct                 REAL NOT NULL,
    thermal_event_flag      INTEGER,
    UNIQUE(vehicle_id, cycle)
);
"""

_CREATE_MAINTENANCE_TICKETS = """
CREATE TABLE IF NOT EXISTS maintenance_tickets (
    ticket_id       TEXT PRIMARY KEY,
    vehicle_id      TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    depot_lat       REAL NOT NULL,
    depot_lon       REAL NOT NULL,
    reason          TEXT NOT NULL,
    technician_id   TEXT NOT NULL
);
"""

_CREATE_COMMANDS = """
CREATE TABLE IF NOT EXISTS commands (
    command_id      TEXT PRIMARY KEY,
    vehicle_id      TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    command_type    TEXT NOT NULL,
    latitude        REAL NOT NULL,
    longitude       REAL NOT NULL,
    ticket_id       TEXT,
    is_attack       INTEGER,
    FOREIGN KEY (ticket_id) REFERENCES maintenance_tickets(ticket_id)
);
"""

# ----------------------------------------------------------------------
# Tier 3 — quarantine / circuit-breaker schema
# ----------------------------------------------------------------------
_CREATE_QUARANTINE = """
CREATE TABLE IF NOT EXISTS vehicle_quarantine (
    vehicle_id      TEXT PRIMARY KEY,
    active          INTEGER NOT NULL DEFAULT 1,
    reason          TEXT NOT NULL,
    action_id       TEXT,
    quarantined_at  TEXT NOT NULL,
    released_at     TEXT,
    released_by     TEXT
);
"""

_CREATE_REJECTED_COMMANDS = """
CREATE TABLE IF NOT EXISTS rejected_commands (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    command_id      TEXT NOT NULL,
    vehicle_id      TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    command_type    TEXT NOT NULL,
    reason          TEXT NOT NULL,
    rejected_at     TEXT NOT NULL
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_telemetry_vehicle ON telemetry(vehicle_id);",
    "CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_tickets_vehicle ON maintenance_tickets(vehicle_id);",
    "CREATE INDEX IF NOT EXISTS idx_commands_vehicle ON commands(vehicle_id);",
    "CREATE INDEX IF NOT EXISTS idx_commands_timestamp ON commands(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_rejected_commands_vehicle ON rejected_commands(vehicle_id);",
]


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_TELEMETRY)
    conn.execute(_CREATE_MAINTENANCE_TICKETS)
    conn.execute(_CREATE_COMMANDS)
    conn.execute(_CREATE_QUARANTINE)
    conn.execute(_CREATE_REJECTED_COMMANDS)
    for stmt in _CREATE_INDEXES:
        conn.execute(stmt)
    conn.commit()


# ----------------------------------------------------------------------
# Insert helpers — telemetry
# ----------------------------------------------------------------------
def _ts(value: datetime) -> str:
    return value.isoformat()


def insert_telemetry(conn: sqlite3.Connection, reading: TelemetryReading) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO telemetry
            (vehicle_id, cycle, timestamp, capacity_kwh, capacity_pct_of_rated,
            rated_capacity_kwh, voltage, temperature_c, soc_pct, is_fast_charge,
            dod_pct, thermal_event_flag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            reading.vehicle_id, reading.cycle, _ts(reading.timestamp),
            reading.capacity_kwh, reading.capacity_pct_of_rated, reading.rated_capacity_kwh,
            reading.voltage, reading.temperature_c, reading.soc_pct,
            int(reading.is_fast_charge), reading.dod_pct,
            None if reading.thermal_event_flag is None else int(reading.thermal_event_flag),
        ),
    )


def insert_telemetry_batch(conn: sqlite3.Connection, readings: Iterable[TelemetryReading]) -> int:
    rows = [
        (
            r.vehicle_id, r.cycle, _ts(r.timestamp), r.capacity_kwh, r.capacity_pct_of_rated,
            r.rated_capacity_kwh, r.voltage, r.temperature_c, r.soc_pct,
            int(r.is_fast_charge), r.dod_pct,
            None if r.thermal_event_flag is None else int(r.thermal_event_flag),
        )
        for r in readings
    ]
    before = conn.total_changes
    conn.executemany(
        """INSERT OR IGNORE INTO telemetry
            (vehicle_id, cycle, timestamp, capacity_kwh, capacity_pct_of_rated,
            rated_capacity_kwh, voltage, temperature_c, soc_pct, is_fast_charge,
            dod_pct, thermal_event_flag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return conn.total_changes - before  # actual rows inserted, excluding duplicates skipped by OR IGNORE


# ----------------------------------------------------------------------
# Insert helpers — maintenance tickets
# ----------------------------------------------------------------------
def insert_maintenance_ticket(conn: sqlite3.Connection, ticket: MaintenanceTicket) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO maintenance_tickets
            (ticket_id, vehicle_id, timestamp, depot_lat, depot_lon, reason, technician_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (ticket.ticket_id, ticket.vehicle_id, _ts(ticket.timestamp),
        ticket.depot_lat, ticket.depot_lon, ticket.reason, ticket.technician_id),
    )


def insert_maintenance_batch(conn: sqlite3.Connection, tickets: Iterable[MaintenanceTicket]) -> int:
    rows = [
        (t.ticket_id, t.vehicle_id, _ts(t.timestamp), t.depot_lat, t.depot_lon,
            t.reason, t.technician_id)
        for t in tickets
    ]
    before = conn.total_changes
    conn.executemany(
        """INSERT OR IGNORE INTO maintenance_tickets
            (ticket_id, vehicle_id, timestamp, depot_lat, depot_lon, reason, technician_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return conn.total_changes - before


# ----------------------------------------------------------------------
# Insert helpers — commands
#
# Both functions now check vehicle_quarantine before writing. An unticketed
# command for a currently-quarantined vehicle is rejected outright (logged
# to rejected_commands instead of commands) — this is the Tier 3 circuit
# breaker actually taking effect at the ingestion boundary, not merely a
# detection flag applied after the write. A ticketed command (a real
# technician visit) always still lands, quarantine or not.
# ----------------------------------------------------------------------
def insert_command(conn: sqlite3.Connection, command: CommandEvent) -> bool:
    """Returns True if the command was written to `commands`, False if it
    was rejected because the vehicle is quarantined and the command has no
    matching maintenance ticket."""
    if command.ticket_id is None and is_vehicle_quarantined(conn, command.vehicle_id):
        log_rejected_command(conn, command, "vehicle is quarantined; command has no maintenance ticket")
        conn.commit()
        return False

    conn.execute(
        """INSERT OR IGNORE INTO commands
            (command_id, vehicle_id, timestamp, command_type, latitude, longitude,
            ticket_id, is_attack)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            command.command_id, command.vehicle_id, _ts(command.timestamp),
            command.command_type.value, command.latitude, command.longitude,
            command.ticket_id,
            None if command.is_attack is None else int(command.is_attack),
        ),
    )
    conn.commit()
    return True


def insert_command_batch(conn: sqlite3.Connection, commands: Iterable[CommandEvent]) -> int:
    """Returns the number of rows actually inserted into `commands` (same
    contract as before: excludes both OR-IGNORE duplicates AND anything
    rejected by quarantine enforcement below)."""
    accepted: List[CommandEvent] = []
    for c in commands:
        if c.ticket_id is None and is_vehicle_quarantined(conn, c.vehicle_id):
            log_rejected_command(conn, c, "vehicle is quarantined; command has no maintenance ticket")
            continue
        accepted.append(c)

    rows = [
        (
            c.command_id, c.vehicle_id, _ts(c.timestamp), c.command_type.value,
            c.latitude, c.longitude, c.ticket_id,
            None if c.is_attack is None else int(c.is_attack),
        )
        for c in accepted
    ]
    before = conn.total_changes
    conn.executemany(
        """INSERT OR IGNORE INTO commands
            (command_id, vehicle_id, timestamp, command_type, latitude, longitude,
            ticket_id, is_attack)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return conn.total_changes - before


# ----------------------------------------------------------------------
# Quarantine CRUD — Tier 3 circuit breaker
# ----------------------------------------------------------------------
def is_vehicle_quarantined(conn: sqlite3.Connection, vehicle_id: str) -> bool:
    row = conn.execute(
        "SELECT active FROM vehicle_quarantine WHERE vehicle_id = ?", (vehicle_id,)
    ).fetchone()
    return bool(row and row["active"])


def set_vehicle_quarantine(
    conn: sqlite3.Connection, vehicle_id: str, reason: str, action_id: Optional[str] = None
) -> dict:
    """Imposes (or re-imposes) quarantine. Idempotent — calling this on an
    already-quarantined vehicle refreshes the reason/action_id rather than
    erroring, since a second high-severity escalation on an
    already-quarantined asset is a normal occurrence, not a bug."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO vehicle_quarantine
            (vehicle_id, active, reason, action_id, quarantined_at, released_at, released_by)
            VALUES (?, 1, ?, ?, ?, NULL, NULL)
            ON CONFLICT(vehicle_id) DO UPDATE SET
            active = 1, reason = excluded.reason, action_id = excluded.action_id,
            quarantined_at = excluded.quarantined_at, released_at = NULL, released_by = NULL""",
        (vehicle_id, reason, action_id, now),
    )
    conn.commit()
    return {"vehicle_id": vehicle_id, "active": True, "reason": reason, "quarantined_at": now}


def release_vehicle_quarantine(
    conn: sqlite3.Connection, vehicle_id: str, released_by: str
) -> Optional[dict]:
    """Returns None if the vehicle wasn't actively quarantined. released_by
    is required — this is the human-accountability trail for a decision
    the agent's LLM is never allowed to make itself (release is deliberately
    excluded from agent/prompts.py's ACTION_TYPES and agent/actions.py's
    ACTION_DISPATCH)."""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """UPDATE vehicle_quarantine SET active = 0, released_at = ?, released_by = ?
            WHERE vehicle_id = ? AND active = 1""",
        (now, released_by, vehicle_id),
    )
    conn.commit()
    if cur.rowcount == 0:
        return None
    return {"vehicle_id": vehicle_id, "active": False, "released_at": now, "released_by": released_by}


def get_quarantine_status(conn: sqlite3.Connection, vehicle_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM vehicle_quarantine WHERE vehicle_id = ?", (vehicle_id,)
    ).fetchone()


def get_active_quarantines(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM vehicle_quarantine WHERE active = 1 ORDER BY quarantined_at DESC"
    ).fetchall()


def log_rejected_command(conn: sqlite3.Connection, command: CommandEvent, reason: str) -> None:
    conn.execute(
        """INSERT INTO rejected_commands
                (command_id, vehicle_id, timestamp, command_type, reason, rejected_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
        (command.command_id, command.vehicle_id, _ts(command.timestamp),
        command.command_type.value, reason, datetime.now(timezone.utc).isoformat()),
    )


def get_rejected_commands(conn: sqlite3.Connection, vehicle_id: Optional[str] = None) -> List[sqlite3.Row]:
    if vehicle_id:
        return conn.execute(
            "SELECT * FROM rejected_commands WHERE vehicle_id = ? ORDER BY rejected_at DESC",
            (vehicle_id,),
        ).fetchall()
    return conn.execute("SELECT * FROM rejected_commands ORDER BY rejected_at DESC").fetchall()


# ----------------------------------------------------------------------
# Query helpers — consumed by models/ (Phase 3) and dashboard/ (Phase 5)
# ----------------------------------------------------------------------
def get_all_vehicle_ids(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT DISTINCT vehicle_id FROM telemetry ORDER BY vehicle_id").fetchall()
    return [r["vehicle_id"] for r in rows]


def get_telemetry_for_vehicle(conn: sqlite3.Connection, vehicle_id: str) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM telemetry WHERE vehicle_id = ? ORDER BY cycle", (vehicle_id,)
    ).fetchall()


def get_tickets_for_vehicle(conn: sqlite3.Connection, vehicle_id: str) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM maintenance_tickets WHERE vehicle_id = ? ORDER BY timestamp", (vehicle_id,)
    ).fetchall()


def get_commands_for_vehicle(conn: sqlite3.Connection, vehicle_id: str) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM commands WHERE vehicle_id = ? ORDER BY timestamp", (vehicle_id,)
    ).fetchall()


def get_unticketed_commands(conn: sqlite3.Connection, vehicle_id: Optional[str] = None) -> List[sqlite3.Row]:
    """Commands with no matching maintenance ticket — the core security-anomaly
    precondition described in the project doc (section 7.1)."""
    if vehicle_id:
        return conn.execute(
            "SELECT * FROM commands WHERE vehicle_id = ? AND ticket_id IS NULL ORDER BY timestamp",
            (vehicle_id,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM commands WHERE ticket_id IS NULL ORDER BY timestamp"
    ).fetchall()


def row_counts(conn: sqlite3.Connection) -> dict:
    return {
        "telemetry": conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0],
        "maintenance_tickets": conn.execute("SELECT COUNT(*) FROM maintenance_tickets").fetchone()[0],
        "commands": conn.execute("SELECT COUNT(*) FROM commands").fetchone()[0],
    }


if __name__ == "__main__":
    # Standalone smoke test: generate fresh simulator data, validate it through
    # schemas.py, load it into a scratch SQLite DB, and confirm round-trip counts.
    from simulator.config import SimulatorConfig
    from simulator.telemetry_generator import TelemetryGenerator
    from simulator.maintenance_generator import MaintenanceGenerator
    from simulator.attack_injector import AttackInjector

    cfg = SimulatorConfig(fleet_size=5, num_cycles=20, random_seed=123)
    tgen = TelemetryGenerator(cfg)
    telem_df = tgen.generate_fleet()
    bounds = tgen.get_vehicle_time_bounds(telem_df)

    mgen = MaintenanceGenerator(cfg)
    tickets_df = mgen.generate_fleet_tickets(bounds)

    ainj = AttackInjector(cfg)
    commands_df = ainj.generate_command_stream(bounds, tickets_df)

    readings = [TelemetryReading(**row) for row in telem_df.to_dict(orient="records")]
    tickets = [MaintenanceTicket(**row) for row in tickets_df.to_dict(orient="records")]
    commands = []
    for row in commands_df.to_dict(orient="records"):
        import math
        if isinstance(row.get("ticket_id"), float) and math.isnan(row["ticket_id"]):
            row["ticket_id"] = None
        commands.append(CommandEvent(**row))

    test_db_path = os.path.join("data", "voltsentinel_test.db")
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    conn = get_connection(test_db_path)
    init_db(conn)

    insert_telemetry_batch(conn, readings)
    insert_maintenance_batch(conn, tickets)
    insert_command_batch(conn, commands)

    print("Row counts after load:", row_counts(conn))
    print("Vehicles in DB:", get_all_vehicle_ids(conn))
    print("Unticketed commands:", len(get_unticketed_commands(conn)))
    conn.close()