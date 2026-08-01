"""
Usage (run from the project root, as a module so `ingestion`/`simulator` are importable):
    python -m scripts.live_feed
    python -m scripts.live_feed --interval 10 --fleet-size 50 --seed 42
    python -m scripts.live_feed --max-ticks 5   # smoke test, then exit
"""

from __future__ import annotations

import argparse
import os

from ingestion.db import get_connection, init_db
from simulator.config import SimulatorConfig
from simulator.live_feed import LiveTelemetryFeed


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously append live telemetry to VoltSentinel's DB")
    parser.add_argument("--db-path", default=os.path.join("data", "voltsentinel.db"))
    parser.add_argument("--fleet-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="must match the seed the DB was originally seeded with")
    parser.add_argument("--interval", type=float, default=15.0, help="seconds between ticks")
    parser.add_argument("--max-ticks", type=int, default=None, help="stop after N ticks (default: run forever)")
    args = parser.parse_args()

    cfg = SimulatorConfig(fleet_size=args.fleet_size, random_seed=args.seed)
    conn = get_connection(args.db_path)
    init_db(conn)

    feed = LiveTelemetryFeed(cfg)
    print(
        f"Starting live telemetry feed against {args.db_path} "
        f"(fleet_size={cfg.fleet_size}, interval={args.interval}s). Ctrl+C to stop."
    )
    try:
        feed.run_forever(conn, interval_seconds=args.interval, max_ticks=args.max_ticks)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
