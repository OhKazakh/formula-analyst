import pandas as pd
import pytest

from src import analysis


@pytest.fixture
def laps() -> pd.DataFrame:
    rows = []
    for driver, stints in {"AAA": [("MEDIUM", 10), ("HARD", 12)], "BBB": [("HARD", 22)]}.items():
        lap_number = 1
        for stint_number, (compound, length) in enumerate(stints, start=1):
            for tyre_life in range(1, length + 1):
                rows.append(
                    {
                        "Driver": driver,
                        "LapNumber": float(lap_number),
                        "Stint": float(stint_number),
                        "Compound": compound,
                        "TyreLife": float(tyre_life),
                        "LapTimeSeconds": 90.0 + 0.1 * tyre_life,
                    }
                )
                lap_number += 1
    return pd.DataFrame(rows)


def test_stint_summary_one_row_per_stint(laps):
    stints = analysis.stint_summary(laps)

    assert stints[["Driver", "Stint"]].values.tolist() == [["AAA", 1], ["AAA", 2], ["BBB", 1]]
    assert stints["Laps"].tolist() == [10, 12, 22]
    assert stints.loc[1, ["StartLap", "EndLap"]].tolist() == [11, 22]


def test_degradation_recovers_slope(laps):
    deg = analysis.degradation(laps, min_laps=5)

    assert len(deg) == 3
    assert deg["DegPerLap"].tolist() == pytest.approx([0.1, 0.1, 0.1])


def test_degradation_skips_short_stints(laps):
    deg = analysis.degradation(laps, min_laps=11)

    assert set(zip(deg["Driver"], deg["Stint"], strict=True)) == {("AAA", 2), ("BBB", 1)}


def test_degradation_empty_when_nothing_qualifies(laps):
    deg = analysis.degradation(laps, min_laps=100)

    assert deg.empty
    assert list(deg.columns) == ["Driver", "Stint", "Compound", "Laps", "DegPerLap"]


def test_fuel_correction_removes_linear_fuel_gain():
    laps = pd.DataFrame({"LapNumber": [1.0, 2.0, 3.0], "LapTimeSeconds": [91.0, 90.95, 90.9]})

    corrected = analysis.fuel_corrected(laps, total_laps=3, fuel_effect=0.05)

    assert corrected.tolist() == pytest.approx([90.9, 90.9, 90.9])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (pd.Timedelta(milliseconds=83226), "1:23.226"),
        (pd.Timedelta(milliseconds=59500), "0:59.500"),
        (pd.NaT, ""),
    ],
)
def test_format_lap_time(value, expected):
    assert analysis.format_lap_time(value) == expected
