from __future__ import annotations

import argparse
import gc
from datetime import date

import fastf1
from fastf1.exceptions import ErgastError, RateLimitExceededError

from src import races, seasons


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.sync",
        description="Download races and championship results and save them to the bundle.",
    )
    parser.add_argument(
        "year", type=int, nargs="?", default=date.today().year, help="Defaults to this season."
    )
    parser.add_argument(
        "--event",
        action="append",
        help="Grand Prix to save, e.g. 'Monza'. Repeatable. Defaults to every completed race.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite races already saved.")
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Only refresh the championship results, without downloading races.",
    )
    return parser.parse_args(argv)


def sync_races(year: int, names: list[str] | None, force: bool) -> int:
    if names:
        events = [fastf1.get_event(year, name)["EventName"] for name in names]
    else:
        events = races.race_calendar(year)["EventName"].tolist()

    saved = races.saved_races()
    already_saved = set(saved.loc[saved["Year"] == year, "EventName"])

    failures = 0
    for event in events:
        if event in already_saved and not force:
            print(f"skip   {year} {event}")
            continue
        try:
            path = races.save_race(races.load_live_race(year, event))
        except races.RaceDataUnavailable as exc:
            failures += 1
            print(f"failed {year} {event}: {exc}")
            continue
        print(f"saved  {path.relative_to(races.ROOT)}", flush=True)
        gc.collect()
    return failures


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fastf1.set_log_level("ERROR")
    races.enable_cache()
    fastf1.Cache.set_disabled()

    try:
        failures = 0 if args.results_only else sync_races(args.year, args.event, args.force)
        path = seasons.save_season(seasons.fetch_season(args.year))
    except (RateLimitExceededError, ErgastError) as exc:
        print(f"stopped: {exc}. Run the same command again later to continue.")
        return 1
    print(f"saved  {path.relative_to(races.ROOT)} championship results", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
