# Formula Analyst

Race analysis dashboard built on [FastF1](https://github.com/theOehrly/Fast-F1) timing data. Pick any Grand Prix since 2018 and see how the race played out: lap-by-lap pace, tyre strategies, fastest-lap telemetry and tyre degradation.

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

The first time a race is opened, its timing data is downloaded (about a minute) and cached in `cache/`. Later loads read from the cache.

## Development

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
ruff format --check .
```

```
app.py             Streamlit UI
src/analysis.py    loading, lap filtering, stint and degradation analysis, charts
tests/             unit tests for the analysis functions
```

`src/analysis.py` has no Streamlit dependency, so the analysis can be reused from scripts or notebooks.

## Methodology

- **Fuel correction** subtracts `fuel_effect × laps remaining` from each lap time, normalising every lap to an empty tank. The default is 0.055 s per lap of fuel and can be changed in the app.
- **Degradation** is the slope of a least-squares line through fuel-corrected lap time against tyre age, fitted per stint. Stints shorter than the configurable minimum (8 laps by default) are skipped.
- A straight line is a simplification: tyre warm-up at the start of a stint and the drop-off at the end are averaged into one number.

## Data and license

Timing data is provided by FastF1. This is an unofficial, non-commercial project and is not affiliated with any racing series or its rights holders.

Released under the [MIT License](LICENSE).
