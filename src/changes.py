"""Commit message for the data the weekly update saved, from the paths git reports as added."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path


def describe(paths: list[Path], year: int | None = None) -> str:
    weekends: dict[str, set[str]] = {}
    for path in paths:
        if path.name != "race.json" or not path.exists():
            continue
        meta = json.loads(path.read_text())
        weekends.setdefault(f"{meta['year']} {meta['event']}", set()).add(
            meta.get("session", "Race")
        )
    if not weekends:
        return f"Update the {year or date.today().year} championship"
    parts = [
        f"the {name}" if "Race" in sessions else f"the {name} sessions so far"
        for name, sessions in weekends.items()
    ]
    return "Add " + (parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1])


def main() -> int:
    print(describe([Path(line.strip()) for line in sys.stdin if line.strip()]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
