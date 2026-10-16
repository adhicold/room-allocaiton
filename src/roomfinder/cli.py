"""Command-line entry point: ``roomfinder evaluate`` and ``roomfinder recommend``."""

from __future__ import annotations

import argparse

import pandas as pd

from .data import load_campus
from .evaluate import run, term_bounds
from .matcher import Request, recommend, suggest_alternatives
from .occupancy import build_occupancy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roomfinder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evaluate", help="run all models and write reports")
    ev.add_argument("--data-dir", default="data/raw")
    ev.add_argument("--out", default="reports")

    rc = sub.add_parser("recommend", help="recommend rooms for one request")
    rc.add_argument("--data-dir", default="data/raw")
    rc.add_argument("--date", required=True)
    rc.add_argument("--start-hour", type=int, required=True)
    rc.add_argument("--duration", type=int, default=1)
    rc.add_argument("--capacity", type=int, required=True)
    rc.add_argument("--projector", action="store_true")
    rc.add_argument("--computers", action="store_true")
    rc.add_argument("--accessible", action="store_true")
    rc.add_argument("--building")
    rc.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    if args.cmd == "evaluate":
        run(args.data_dir, args.out)
        print(f"Wrote metrics and reports to {args.out}/")
        return 0

    campus = load_campus(args.data_dir)
    lo, hi = term_bounds(campus)
    occ = build_occupancy(campus.rooms, campus.schedule, campus.events, lo, hi)
    req = Request(pd.Timestamp(args.date), args.start_hour, args.duration, args.capacity,
                  args.projector, args.computers, args.accessible, args.building)
    recs = recommend(campus.rooms, occ, req, top_k=args.top_k)
    if not recs:
        print("No feasible room. Nearest alternatives:")
        for alt in suggest_alternatives(campus.rooms, occ, req):
            print(f"  start {alt['start_hour']:02d}:00 ({alt['shift_hours']:+d}h): "
                  f"{alt['room'].room_id}")
        return 1
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r.room_id} [{r.building}, {r.capacity} seats] score={r.score:.3f}")
        for reason in r.reasons:
            print(f"     - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
