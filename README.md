# Formula Analyst

Race analysis dashboard built on [FastF1](https://github.com/theOehrly/Fast-F1) timing data. Pick a Grand Prix and see how the race played out, or follow a whole championship season. Every race from 2018 to the latest 2026 round is included.

**[Open the live app](https://formula-analyst.streamlit.app/)**

![Race replay of the 2024 Italian Grand Prix](docs/replay.gif)

## Features

**Race**

Every race opens with a summary: the podium, the fastest lap, the number of pit stops and any safety car, virtual safety car or red flag. The analysis is split into five tabs.

- **Overview.** Every driver's position lap by lap, with selected drivers highlighted, plus who led and for how many laps.
- **Replay.** Every car on a map of the circuit through the whole race, with the running order alongside. Play it or drag the slider lap by lap.
- **Pace.** Lap times for any set of drivers, optionally limited to representative laps (no pit in/out laps, no laps under safety car or flags, no laps FastF1 marks as inaccurate) and fuel-corrected. Each team's lap time spread ordered by median pace, and each driver's lap time distribution coloured by tyre compound.
- **Strategy.** Every driver's stints, coloured by compound and ordered by finishing position, and the degradation rate of each stint from a linear fit of fuel-corrected lap time against tyre age, summarised by compound. A machine learning model trained on every other race since 2019 predicts how much time each compound should lose at that circuit, shown against the race's actual laps.
- **Telemetry.** Speed against distance for two drivers' fastest laps, with corner markers, and the same laps on a track map coloured by speed or by who was quicker in each mini-sector.

**Championship**

- Standings after any round, with each driver's maximum possible points and whether they can still win the title.
- Points progression through the season.
- Points per round for every driver, race and sprint combined.

Every view has its own link, so a race, a tab or a point in a championship can be shared directly, for example the [2023 British Grand Prix strategy](https://formula-analyst.streamlit.app/?season=2023&race=british-grand-prix&tab=strategy) or the [2021 standings after Silverstone](https://formula-analyst.streamlit.app/championship?season=2021&round=10). The app follows the system's light or dark setting, and drivers are drawn in their teams' official colours. Every chart is interactive: hover a point for its lap time, tyre, position or speed.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/overview-dark.png">
  <img alt="Race overview" src="docs/overview.png">
</picture>

| Pace | Team pace |
| --- | --- |
| ![Pace](docs/pace.png) | ![Team pace](docs/team-pace.png) |
| **Strategy** | **Tyre model** |
| ![Strategy](docs/strategy.png) | ![Tyre model](docs/tyre-model.png) |
| **Telemetry** | **Championship** |
| ![Telemetry](docs/telemetry.png) | ![Championship](docs/championship.png) |

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

After a race weekend, one command adds the new races, updates the championship and retrains the tyre model:

```bash
pip install -r requirements-dev.txt
python -m src.sync
```

It defaults to the current season. Other forms:

```bash
python -m src.sync 2023 --event Monza --event Silverstone --force
python -m src.sync 2026 --results-only
```

Saved races are skipped unless `--force` is passed, and `--results-only` updates the championship without downloading races. The sync has to run on a connection the live timing service accepts; it rejects GitHub's servers as well as Streamlit's, so it can't run as a scheduled workflow. FastF1 allows 500 requests an hour, roughly 20 races; when a limit is reached the command stops and can be run again later to continue. Set `FORMULA_ANALYST_OFFLINE=1` to run the app the way it runs when deployed, with bundled races only.

## Development

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
ruff format --check .
python -m src.model
```

The last command retrains the tyre model on the bundled races and writes its curves and metrics to `data/tyre-model/`. scikit-learn is only needed for that step, so the deployed app doesn't install it.

```
app.py                  page navigation
views/race.py           race page
views/championship.py   championship page
views/data.py           cached data loading for both pages
views/theme.py          colours and styles shared by the pages
views/links.py          shareable links from URL parameters
src/analysis.py         race analysis and charts
src/championship.py     standings, title maths and championship charts
src/races.py            loading races from FastF1 or the bundle, saving bundles
src/seasons.py          loading and saving season results
src/degradation.py      time lost to tyre wear in each stint, and the saved model curves
src/model.py            training and cross-validating the tyre model
src/sync.py             command line tool that builds the bundle
data/                   bundled races and season results
tests/                  unit tests
```

Everything in `src/` works on plain pandas DataFrames and has no Streamlit dependency; only `src/races.py`, `src/seasons.py` and `src/sync.py` use FastF1, and only `src/model.py` uses scikit-learn.

## Methodology

- **Race summary.** Safety car and virtual safety car periods are counted in the winner's laps that ran under them. Pit stops leave out pit lane entries during a red flag, when every car waits in the pit lane. Laps led and lead changes come from the running order at the end of each lap.
- **Fuel correction** subtracts `fuel_effect × laps remaining` from each lap time, normalising every lap to an empty tank. The default is 0.055 s per lap of fuel and can be changed in the app.
- **Degradation** is the slope of a least-squares line through fuel-corrected lap time against tyre age, fitted per stint. Stints shorter than the configurable minimum (8 laps by default) are skipped.
- A straight line is a simplification: tyre warm-up at the start of a stint and the drop-off at the end are averaged into one number.
- **Tyre model.** For every dry stint from 2019 onwards, when the compounds were named soft, medium and hard, each representative lap's fuel-corrected time is compared with the median of the stint's first three laps. That gives the time lost since the stint began, which includes tyre wear and the track rubbering in. A gradient-boosted tree model (scikit-learn's `HistGradientBoostingRegressor`) predicts it from the compound, tyre age, the age of the set when the stint began, the circuit and the season. It is trained on 126,491 laps from 6,458 stints in 161 races. Cross-validated in five folds grouped by race, so every race is predicted by a model that never saw it, it is off by 0.58 s a lap on average, against 0.63 s for a straight line per compound and 0.75 s for assuming no wear. Lap-to-lap variation from traffic and driving limits how close any model can get. The curves shown for each race come from the fold that held it out.
- **Positions** are the order in which drivers completed each lap. They match FastF1's own position data.
- **Pace comparison** uses representative laps within 107% of the fastest one, the same quick-lap threshold as FastF1.
- **Mini-sectors** split the fastest lap into 25 equal distances. A driver's time through each one comes from integrating their speed trace.
- **Race replay** positions come from each lap's start and end times. Within a lap, cars follow the speed profile of the winner's fastest lap, so they slow down in corners rather than moving at constant speed. The order shown is the order on track, before any penalties. Track outlines come from the MultiViewer circuit data that FastF1 uses; the 2020 Sakhir and 2026 Spanish Grand Prix have none.
- **Championship standings** add up race and sprint points after each round and break ties by countback over classified finishes. They match the official standings for every bundled season.
- **Who can still win** compares the leader's points with each driver's points plus the most still available: a race win (and the fastest-lap point from 2019 to 2024) for every remaining round, plus a sprint win where there is one.

## Data and license

Timing data is provided by FastF1, and the bundled data is derived from it. Race and sprint results come from the [Jolpica F1 API](https://github.com/jolpica/jolpica-f1), the successor to Ergast, through FastF1. This is an unofficial, non-commercial project and is not affiliated with any racing series or its rights holders.

Released under the [MIT License](LICENSE).
