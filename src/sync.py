from __future__ import annotations

import argparse
import gc

import fastf1
from fastf1.exceptions import RateLimitExceededError

from src import races


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.sync",
        description="Download races with FastF1 and save them to the bundled data directory.",
    )
    parser.add_argument("year", type=int)
    parser.add_argument(
        "--event",
        action="append",
        help="Grand Prix to save, e.g. 'Monza'. Repeatable. Defaults to every completed race.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite races already saved.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fastf1.set_log_level("ERROR")
    races.enable_cache()
    fastf1.Cache.set_disabled()

    if args.event:
        events = [fastf1.get_event(args.year, name)["EventName"] for name in args.event]
    else:
        events = races.race_calendar(args.year)["EventName"].tolist()

    saved = races.saved_races()
    already_saved = set(saved.loc[saved["Year"] == args.year, "EventName"])

    failures = 0
    for event in events:
        if event in already_saved and not args.force:
            print(f"skip   {args.year} {event}")
            continue
        try:
            path = races.save_race(races.load_live_race(args.year, event))
        except races.RaceDataUnavailable as exc:
            failures += 1
            print(f"failed {args.year} {event}: {exc}")
            continue
        except RateLimitExceededError as exc:
            print(f"stopped: {exc}. Run the same command again later to continue.")
            return 1
        print(f"saved  {path.relative_to(races.ROOT)}", flush=True)
        gc.collect()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
