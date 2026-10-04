import json

import pandas as pd
import pytest

from src import degradation, model, races, seasons
from tests.test_degradation import TOTAL_LAPS, stint_laps

WEAR = {"SOFT": 0.12, "MEDIUM": 0.06, "HARD": 0.02}


def synthetic_race(round_number: int) -> races.Race:
    laps = pd.concat(
        [
            stint_laps(f"D{index}", compound, range(2, 22), wear)
            for index, (compound, wear) in enumerate(WEAR.items())
        ],
        ignore_index=True,
    )
    laps["Team"] = "Team"
    laps["LapTime"] = pd.to_timedelta(laps["LapTimeSeconds"], unit="s")
    laps["IsPersonalBest"] = False
    return races.Race(
        year=2024,
        round_number=round_number,
        event=f"Race {round_number}",
        total_laps=TOTAL_LAPS,
        order=list(laps["Driver"].unique()),
        laps=laps,
        telemetry=pd.DataFrame(columns=races.TELEMETRY_COLUMNS),
        corners=pd.DataFrame(columns=races.CORNER_COLUMNS),
        driver_styles={},
        compound_colors={},
    )


@pytest.fixture
def bundle(tmp_path):
    rounds = range(1, 7)
    for round_number in rounds:
        races.save_race(synthetic_race(round_number), tmp_path)
    schedule = pd.DataFrame(
        {
            "Round": list(rounds),
            "EventName": [f"Race {r}" for r in rounds],
            "Country": "Italy",
            "Location": ["Monza", "Imola"] * 3,
            "Sprint": False,
            "Date": pd.Timestamp("2024-09-01"),
        }
    )
    results = pd.DataFrame(columns=seasons.RESULT_COLUMNS)
    seasons.save_season(seasons.Season(2024, schedule, results), tmp_path)
    return tmp_path


def test_build_saves_held_out_curves_and_metrics(bundle):
    path = model.build(bundle)

    metrics = json.loads((path / "metrics.json").read_text())
    curves = degradation.load_curves(bundle)
    assert (metrics["races"], metrics["stints"], metrics["first_season"]) == (6, 18, 2024)
    assert metrics["mae"]["model"] < metrics["mae"]["no_wear"]
    assert curves["Race"].nunique() == 6
    after = degradation.loss_after(degradation.race_curves(curves, 2024, 3), laps=12)
    assert after["SOFT"] > after["MEDIUM"] > after["HARD"]


def test_curve_grid_starts_on_a_fresh_set():
    bundled = pd.DataFrame(
        {"Race": ["2024-01"], "Year": [2024], "Location": ["Monza"], "TrackTemp": [40.0]}
    )
    table = pd.DataFrame({"Compound": ["SOFT"] * 20, "TyreLife": range(1, 21)})

    grid = model.curve_grid(bundled, table)

    assert grid["Compound"].unique().tolist() == ["SOFT"]
    assert grid["TyreLife"].min() == degradation.FRESH_TYRE_AGE
    assert grid["StartAge"].unique().tolist() == [degradation.FRESH_TYRE_AGE]
