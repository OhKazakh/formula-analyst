import pandas as pd
import pytest

from src import analysis, degradation

TOTAL_LAPS = 30


def stint_laps(
    driver: str = "AAA",
    compound: str = "MEDIUM",
    laps: range = range(2, 14),
    wear: float = 0.1,
    stint: float = 1.0,
) -> pd.DataFrame:
    no_time = pd.Series([pd.NaT] * len(laps), dtype="timedelta64[ns]")
    numbers = pd.Series(laps, dtype=float)
    fuel = analysis.DEFAULT_FUEL_EFFECT * (TOTAL_LAPS - numbers)
    return pd.DataFrame(
        {
            "Driver": driver,
            "LapNumber": numbers,
            "LapTimeSeconds": 90.0 + wear * numbers + fuel,
            "Stint": stint,
            "Compound": compound,
            "TyreLife": numbers,
            "PitInTime": no_time,
            "PitOutTime": no_time,
            "TrackStatus": "1",
            "IsAccurate": True,
        }
    )


def test_stint_losses_measure_wear_from_the_opening_laps():
    losses = degradation.stint_losses(stint_laps(), TOTAL_LAPS)

    assert losses["StartAge"].unique().tolist() == [2.0]
    expected = 0.1 * (losses["TyreLife"] - 3.0)
    assert losses["Loss"].tolist() == pytest.approx(expected.tolist())


def test_stint_losses_skip_short_and_wet_stints():
    laps = pd.concat(
        [
            stint_laps("AAA", "SOFT", range(2, 6)),
            stint_laps("BBB", "INTERMEDIATE", range(2, 14)),
            stint_laps("CCC", "HARD", range(2, 14)),
        ]
    )

    losses = degradation.stint_losses(laps, TOTAL_LAPS)

    assert losses["Driver"].unique().tolist() == ["CCC"]
    assert losses["Compound"].unique().tolist() == ["HARD"]


def test_stint_losses_empty_without_long_stints():
    losses = degradation.stint_losses(stint_laps(laps=range(2, 5)), TOTAL_LAPS)

    assert losses.empty
    assert list(losses.columns) == degradation.LOSS_COLUMNS


def curves() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Race": ["2024-16"] * 4 + ["2024-17"],
            "Compound": ["SOFT", "SOFT", "HARD", "HARD", "SOFT"],
            "TyreLife": [2.0, 17.0, 2.0, 17.0, 17.0],
            "Predicted": [0.0, 0.8, 0.0, 0.3, 9.9],
        }
    )


def test_race_curves_and_loss_after_the_horizon():
    race = degradation.race_curves(curves(), 2024, 16)

    assert len(race) == 4
    assert degradation.loss_after(race) == {"SOFT": pytest.approx(0.8), "HARD": pytest.approx(0.3)}


def test_missing_model_files(tmp_path):
    assert degradation.load_curves(tmp_path).empty
    assert degradation.load_metrics(tmp_path) is None
