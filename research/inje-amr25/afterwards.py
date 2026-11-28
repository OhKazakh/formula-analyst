"""What the AMR25 did at Inje, against the prediction.

On 28 November 2026 the car ran Course A (2.577 km), not the 3.908 km full course predict.py
models. This builds Course A from the full course and the A Course link in OpenStreetMap, reruns
the straight, the top speed and the computer lap on it, and carries the lap-record bridge across
with the computer lap's Course A to full course ratio.

Run from the repository root: python research/inje-amr25/afterwards.py
It prints the comparison and writes afterwards.json. The A Course link is cached in inje_osm_a.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import predict as p  # noqa: E402

# https://www.autoracing.co.kr/news/articleView.html?idxno=50652
COURSE_A_LENGTH = 2577.0
TIMED_LAPS = (55.438, 54.493, 54.108, 53.675)
TOP_SPEED_RUNS = (242.0, 272.0, 271.0, 273.0)

# https://www.openstreetmap.org/way/677355824
OSM_A_LINK = 677355824
OSM_A_FILE = HERE / "inje_osm_a.json"


def a_link() -> list[dict]:
    if OSM_A_FILE.exists():
        return json.loads(OSM_A_FILE.read_text())["a_course_link"]
    response = requests.post(
        p.OVERPASS,
        data={"data": f"[out:json][timeout:60];way({OSM_A_LINK});out geom;"},
        headers={"User-Agent": "formula-analyst-research"},
        timeout=90,
    )
    response.raise_for_status()
    link = response.json()["elements"][0]["geometry"]
    cached = {"source": "© OpenStreetMap contributors, ODbL 1.0", "a_course_link": link}
    OSM_A_FILE.write_text(json.dumps(cached, indent=1) + "\n")
    return link


def course_a(course: list[dict], link: list[dict]) -> list[dict]:
    start, end = course.index(link[0]), course.index(link[-1])
    lap = len(course) - 1  # the way is closed, so its last node is its first
    # The link closes two loops. Course A is the one round the start of the way, past the pits.
    return [course[(end + i) % lap] for i in range((start - end) % lap + 1)] + link[1:]


def ranges_on(scale: float, gap: float) -> dict[str, tuple[float, float]]:
    fit, ratio, era = p.fit_bridge(p.BRIDGE), p.class_ratio(p.BRIDGE), p.era_factor()
    full = {}
    for pace, text in p.F1_KIC_2013.items():
        result = p.predict(p.seconds(text), fit, ratio, era)
        full[pace] = (min(result.values()), max(result.values()))
    middle = sum(full["qualifying"]) / 2
    quali = tuple(lap * scale for lap in full["qualifying"])
    # The demo judgement keeps its share of the lap: 8 to 15 s on a full-course lap.
    slower = [s / middle for s in p.DEMO_SLOWER]
    return {
        "pole_level": quali,
        "amr25": tuple(lap * (1 + gap) for lap in quali),
        "race": tuple(lap * scale for lap in full["race"]),
        "demo": tuple(sum(quali) / 2 * (1 + s) for s in slower),
    }


def main() -> int:
    geometry = p.osm_geometry()
    layouts = {
        "full": (geometry, p.INJE_LENGTH),
        "a": ({**geometry, "course": course_a(geometry["course"], a_link())}, COURSE_A_LENGTH),
    }
    loaded = p.load_amr25()
    samples = [s for w in loaded for d in p.AMR25 for s in p.corner_samples(w["race"], d)]
    corners = p.fit_corners(samples)

    courses = {}
    for name, (layout, length) in layouts.items():
        straight = p.main_straight(layout, length)
        computer = p.computer_lap(layout, loaded, corners, length)
        courses[name] = {
            "length_m": length,
            "osm_length_m": straight["osm_length_m"],
            "straight": straight,
            "top_speed_kmh": p.top_speed(straight, loaded, corners)["range_kmh"],
            "computer_lap": computer["simulated"],
            "gap_to_pole": computer["gap_to_pole"],
        }
        print(
            f"{name:>4}: {length:.0f} m, main straight {straight['straight_m']:.0f} m after a "
            f"{straight['corner_radius_m']:.0f} m corner, computer lap "
            f"{p.lap_time(computer['simulated'])}"
        )

    full, a = courses["full"], courses["a"]
    scale = a["computer_lap"] / full["computer_lap"]
    ranges = ranges_on(scale, full["gap_to_pole"])
    best = min(TIMED_LAPS)
    print(f"\nCourse A takes {scale:.3f} of a full-course lap in the simulator")
    for name, (low, high) in ranges.items():
        print(f"  {name:<10} {low:.1f} to {high:.1f} s")
    print(f"  actual     {best:.3f} s, best of {', '.join(f'{lap:.3f}' for lap in TIMED_LAPS)}")
    low, high = a["top_speed_kmh"]
    print(f"\nTop speed on Course A: {low:.0f} to {high:.0f} km/h, actual {max(TOP_SPEED_RUNS):g}")

    results = {
        "courses": courses,
        "scale": scale,
        "ranges": ranges,
        "actual": {"timed_laps": TIMED_LAPS, "top_speed_runs": TOP_SPEED_RUNS},
    }
    (HERE / "afterwards.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\nWrote afterwards.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
