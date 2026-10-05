from __future__ import annotations

import argparse
import gc
from datetime import date

import fastf1
import pandas as pd
from fastf1.exceptions import ErgastError, RateLimitExceededError

from src import model, races, seasons, team_logos


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
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Download saved races again if they were saved in an older format.",
    )
    parser.add_argument(
        "--sessions",
        action="store_true",
        help="Also download practice, qualifying and sprint sessions.",
    )
    parser.add_argument(
        "--session",
        action="append",
        choices=[name for name in races.SESSIONS if name != races.RACE],
        help="Only download these sessions, e.g. 'Qualifying'. Repeatable.",
    )
    parser.add_argument(
        "--skip-model", action="store_true", help="Don't retrain the tyre model afterwards."
    )
    return parser.parse_args(argv)


def outdated_events(year: int) -> list[str]:
    saved = races.saved_races()
    saved = saved[saved["Year"] == year]
    return [
        event
        for event, path in zip(saved["EventName"], saved["Path"], strict=True)
        if races.bundle_format(path) < races.FORMAT_VERSION
    ]


def sync_races(year: int, names: list[str] | None, force: bool, upgrade: bool = False) -> int:
    if upgrade:
        events, force = outdated_events(year), True
    elif names:
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
        except (RateLimitExceededError, ErgastError):
            raise
        except Exception as exc:
            failures += 1
            print(f"failed {year} {event}: {exc}")
            continue
        print(f"saved  {path.relative_to(races.ROOT)}", flush=True)
        gc.collect()
    return failures


def weekend_sessions(schedule: pd.DataFrame, event: str) -> list[str]:
    weekend = schedule[schedule["EventName"] == event].iloc[0]
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return [
        weekend[f"Session{number}"]
        for number in range(1, 6)
        if weekend[f"Session{number}"] in races.SESSIONS
        and weekend[f"Session{number}"] != races.RACE
        and weekend[f"Session{number}DateUtc"] < now
    ]


def sync_sessions(
    year: int, names: list[str] | None, force: bool, only: list[str] | None = None
) -> int:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if names:
        events = [fastf1.get_event(year, name)["EventName"] for name in names]
    else:
        events = races.race_calendar(year)["EventName"].tolist()
    saved = races.saved_sessions()
    saved = saved[saved["Year"] == year]
    already_saved = set(zip(saved["EventName"], saved["Session"], strict=True))

    failures = 0
    for event in events:
        for session in weekend_sessions(schedule, event):
            if only and session not in only:
                continue
            if (event, session) in already_saved and not force:
                continue
            try:
                path = races.save_race(races.load_live_race(year, event, session))
            except (RateLimitExceededError, ErgastError):
                raise
            except Exception as exc:
                failures += 1
                print(f"failed {year} {event} {session}: {exc}")
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
        failures = (
            0 if args.results_only else sync_races(args.year, args.event, args.force, args.upgrade)
        )
        if (args.sessions or args.session) and not args.results_only:
            failures += sync_sessions(args.year, args.event, args.force, args.session)
        saved = seasons.load_season(args.year) if args.year in seasons.saved_seasons() else None
        season = seasons.fetch_season(
            args.year, saved.team_standings if saved is not None else None
        )
        path = seasons.save_season(season)
    except (RateLimitExceededError, ErgastError) as exc:
        print(f"stopped: {exc}. Run the same command again later to continue.")
        return 1
    print(f"saved  {path.relative_to(races.ROOT)} championship results", flush=True)
    if logos := team_logos.fetch_logos(args.year, season.results["Team"].unique().tolist()):
        print(f"saved  {args.year} team logos: {', '.join(logos)}", flush=True)
    if not args.skip_model:
        path = model.build()
        print(f"saved  {path.relative_to(races.ROOT)} tyre model", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
