# Formula Analyst

Race analysis dashboard built on [FastF1](https://github.com/theOehrly/Fast-F1) timing data. Pick a Grand Prix and see how the race played out: lap-by-lap pace, tyre strategies, fastest-lap telemetry and tyre degradation.

![Race pace](docs/pace.png)

## Features

- **Race pace.** Lap times for any set of drivers. Optionally limited to representative laps (no pit in/out laps, no laps under safety car or flags, no laps FastF1 marks as inaccurate) and fuel-corrected.
- **Tyre strategy.** Every driver's stints, coloured by compound and ordered by finishing position.
- **Fastest lap.** Speed against distance for two drivers' fastest laps, with corner markers.
- **Degradation.** Per-stint degradation rate from a linear fit of fuel-corrected lap time against tyre age, summarised by compound.

| Tyre strategy | Fastest lap |
| --- | --- |
| ![Tyre strategy](docs/strategy.png) | ![Fastest lap](docs/fastest-lap.png) |

## Running locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Race data

Every race from 2018 to the latest 2026 round ships with the app in `data/` (about 18 MB), as compact Parquet files with the lap table and each driver's fastest-lap speed trace. The one exception is the 2018 Italian Grand Prix, whose tyre data FastF1 can't process. The live timing service rejects requests from many hosting providers, Streamlit Community Cloud included, so the deployed app reads only this bundle.

When the live timing service is reachable, as it usually is on a home connection, the app also downloads races that aren't bundled. The first download of a race takes about a minute and is cached in `cache/`.

To add new races, for example after a race weekend:

```bash
python -m src.sync 2026
python -m src.sync 2023 --event Monza --event Silverstone --force
```

Saved races are skipped unless `--force` is passed. FastF1 allows 500 requests an hour, roughly 20 races; when the limit is reached the command stops and can be run again later to continue. Set `FORMULA_ANALYST_OFFLINE=1` to run the app the way it runs when deployed, with bundled races only.

## Development

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
ruff format --check .
```

```
app.py             Streamlit UI
src/analysis.py    lap filtering, stint and degradation analysis, charts
src/races.py       loading races from FastF1 or the bundle, saving bundles
src/sync.py        command line tool that bundles races
data/              bundled races
tests/             unit tests
```

`src/analysis.py` works on plain pandas DataFrames and has no FastF1 or Streamlit dependency.

## Methodology

- **Fuel correction** subtracts `fuel_effect × laps remaining` from each lap time, normalising every lap to an empty tank. The default is 0.055 s per lap of fuel and can be changed in the app.
- **Degradation** is the slope of a least-squares line through fuel-corrected lap time against tyre age, fitted per stint. Stints shorter than the configurable minimum (8 laps by default) are skipped.
- A straight line is a simplification: tyre warm-up at the start of a stint and the drop-off at the end are averaged into one number.

## Data and license

Timing data is provided by FastF1, and the bundled data is derived from it. This is an unofficial, non-commercial project and is not affiliated with any racing series or its rights holders.

Released under the [MIT License](LICENSE).
