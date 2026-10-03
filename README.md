# Formula Analyst

Race analysis dashboard built on [FastF1](https://github.com/theOehrly/Fast-F1) timing data. Pick a Grand Prix and see how the race played out, or follow a whole championship season. Every race from 2018 to the latest 2026 round is included.

![Race pace](docs/pace.png)

## Features

**Race**

- **Race pace.** Lap times for any set of drivers. Optionally limited to representative laps (no pit in/out laps, no laps under safety car or flags, no laps FastF1 marks as inaccurate) and fuel-corrected.
- **Positions.** Every driver's position lap by lap, with selected drivers highlighted.
- **Race replay.** Every car on a map of the circuit through the whole race, with the running order alongside. Play it or drag the slider lap by lap.
- **Pace comparison.** Each team's lap time spread, ordered by median pace, and each driver's lap time distribution coloured by tyre compound.
- **Tyre strategy.** Every driver's stints, coloured by compound and ordered by finishing position.
- **Fastest lap.** Speed against distance for two drivers' fastest laps, with corner markers, and the same laps on a track map coloured by speed or by who was quicker in each mini-sector.
- **Degradation.** Per-stint degradation rate from a linear fit of fuel-corrected lap time against tyre age, summarised by compound.

**Championship**

- Standings after any round, with each driver's maximum possible points and whether they can still win the title.
- Points progression through the season.
- Points per round for every driver, race and sprint combined.

![Race replay](docs/replay.png)

| Positions | Pace comparison |
| --- | --- |
| ![Positions](docs/positions.png) | ![Pace comparison](docs/comparison.png) |
| **Tyre strategy** | **Fastest lap** |
| ![Tyre strategy](docs/strategy.png) | ![Fastest lap](docs/fastest-lap.png) |

![Championship](docs/championship.png)

## Running locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Race data

Every race from 2018 to the latest 2026 round ships with the app in `data/` (about 24 MB), as compact Parquet files with the lap table, each driver's fastest-lap speed trace and the track outline. The one exception is the 2018 Italian Grand Prix, whose tyre data FastF1 can't process. Each season also has its race and sprint results and its calendar. The live timing service rejects requests from many hosting providers, Streamlit Community Cloud included, so the deployed app reads only this bundle.

When the live timing service is reachable, as it usually is on a home connection, the app also downloads races that aren't bundled. The first download of a race takes about a minute and is cached in `cache/`.

To add new races and update the championship, for example after a race weekend:

```bash
python -m src.sync 2026
python -m src.sync 2023 --event Monza --event Silverstone --force
python -m src.sync 2026 --results-only
```

Saved races are skipped unless `--force` is passed, and `--results-only` updates the championship without downloading races. FastF1 allows 500 requests an hour, roughly 20 races; when a limit is reached the command stops and can be run again later to continue. Set `FORMULA_ANALYST_OFFLINE=1` to run the app the way it runs when deployed, with bundled races only.

## Development

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
ruff format --check .
```

```
app.py                  page navigation
views/race.py           race page
views/championship.py   championship page
views/data.py           cached data loading for both pages
src/analysis.py         race analysis and charts
src/championship.py     standings, title maths and championship charts
src/races.py            loading races from FastF1 or the bundle, saving bundles
src/seasons.py          loading and saving season results
src/sync.py             command line tool that builds the bundle
data/                   bundled races and season results
tests/                  unit tests
```

Everything in `src/` works on plain pandas DataFrames and has no Streamlit dependency; only `src/races.py`, `src/seasons.py` and `src/sync.py` use FastF1.

## Methodology

- **Fuel correction** subtracts `fuel_effect × laps remaining` from each lap time, normalising every lap to an empty tank. The default is 0.055 s per lap of fuel and can be changed in the app.
- **Degradation** is the slope of a least-squares line through fuel-corrected lap time against tyre age, fitted per stint. Stints shorter than the configurable minimum (8 laps by default) are skipped.
- A straight line is a simplification: tyre warm-up at the start of a stint and the drop-off at the end are averaged into one number.
- **Positions** are the order in which drivers completed each lap. They match FastF1's own position data.
- **Pace comparison** uses representative laps within 107% of the fastest one, the same quick-lap threshold as FastF1.
- **Mini-sectors** split the fastest lap into 25 equal distances. A driver's time through each one comes from integrating their speed trace.
- **Race replay** positions come from each lap's start and end times. Within a lap, cars follow the speed profile of the winner's fastest lap, so they slow down in corners rather than moving at constant speed. The order shown is the order on track, before any penalties. Track outlines come from the MultiViewer circuit data that FastF1 uses; the 2020 Sakhir and 2026 Spanish Grand Prix have none.
- **Championship standings** add up race and sprint points after each round and break ties by countback over classified finishes. They match the official standings for every bundled season.
- **Who can still win** compares the leader's points with each driver's points plus the most still available: a race win (and the fastest-lap point from 2019 to 2024) for every remaining round, plus a sprint win where there is one.

## Data and license

Timing data is provided by FastF1, and the bundled data is derived from it. Race and sprint results come from the [Jolpica F1 API](https://github.com/jolpica/jolpica-f1), the successor to Ergast, through FastF1. This is an unofficial, non-commercial project and is not affiliated with any racing series or its rights holders.

Released under the [MIT License](LICENSE).
