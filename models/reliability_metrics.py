"""MTBF/MTTR reliability metrics, computed from agent-flagged maintenance
triggers (agent_actions) and telemetry timestamps — no new ingestion."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class ReliabilityProfile:
    vehicle_id: str
    maintenance_trigger_count: int
    ticket_count: int
    mtbf_hours: Optional[float]
    mttr_hours: Optional[float]
    status: str  # "ok" | "insufficient_data"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _parse_ts(value) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


class ReliabilityAnalyzer:
    def analyze_vehicle(self, conn: sqlite3.Connection, vehicle_id: str) -> ReliabilityProfile:
        from agent.actions import get_actions_for_vehicle
        from ingestion.db import get_telemetry_for_vehicle, get_tickets_for_vehicle

        actions = get_actions_for_vehicle(conn, vehicle_id)
        trigger_ts = sorted(
            _parse_ts(a["created_at"]) for a in actions if a["action_type"] == "maintenance_trigger"
        )
        ticket_count = len(get_tickets_for_vehicle(conn, vehicle_id))

        mtbf_hours = None
        if len(trigger_ts) >= 2:
            gaps = [(b - a).total_seconds() / 3600 for a, b in zip(trigger_ts, trigger_ts[1:])]
            mtbf_hours = round(sum(gaps) / len(gaps), 1)

        mttr_hours = None
        if trigger_ts:
            telemetry_ts = sorted(_parse_ts(r["timestamp"]) for r in get_telemetry_for_vehicle(conn, vehicle_id))
            # a maintenance trigger's "repair time" is approximated as the telemetry
            # gap straddling it — the span between the last reading before the fault
            # was flagged and the first reading once the vehicle is back online.
            repair_gaps = []
            for t in trigger_ts:
                before = [ts for ts in telemetry_ts if ts <= t]
                after = [ts for ts in telemetry_ts if ts > t]
                if before and after:
                    repair_gaps.append((after[0] - before[-1]).total_seconds() / 3600)
            if repair_gaps:
                mttr_hours = round(sum(repair_gaps) / len(repair_gaps), 1)

        status = "ok" if (mtbf_hours is not None or mttr_hours is not None) else "insufficient_data"
        return ReliabilityProfile(
            vehicle_id=vehicle_id,
            maintenance_trigger_count=len(trigger_ts),
            ticket_count=ticket_count,
            mtbf_hours=mtbf_hours,
            mttr_hours=mttr_hours,
            status=status,
        )

    def analyze_fleet(self, conn: sqlite3.Connection) -> pd.DataFrame:
        from ingestion.db import get_all_vehicle_ids

        vehicle_ids = get_all_vehicle_ids(conn)
        if not vehicle_ids:
            return pd.DataFrame(columns=[
                "vehicle_id", "maintenance_trigger_count", "ticket_count",
                "mtbf_hours", "mttr_hours", "status",
            ])
        rows: List[dict] = [self.analyze_vehicle(conn, vid).to_dict() for vid in vehicle_ids]
        return pd.DataFrame(rows)
