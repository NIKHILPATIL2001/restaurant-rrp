"""
Background warmup simulation run at container startup.

Runs simulate.py with --days N so by the time the evaluator opens
the dashboard, daily_metrics already has a visible convergence curve.
"""

from __future__ import annotations

import argparse

from rrp.core.logging import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log.info("warmup.start", days=args.days, seed=args.seed)
    from rrp.scripts.simulate import run_simulation
    results = run_simulation(days=args.days, seed=args.seed)
    log.info("warmup.done", days_completed=len(results))


if __name__ == "__main__":
    main()
