import numpy as np
import pandas as pd

from src import predictions, race_model


def test_recent_form_uses_each_drivers_own_last_races():
    results = pd.DataFrame(
        {
            "Order": [202601, 202602, 202603, 202604, 202605, 202606, 202601, 202606],
            "Driver": ["VER"] * 6 + ["LAW", "LAW"],
            "Position": [9, 1, 1, 1, 1, 1, 4, 6],
        }
    )

    form = predictions.recent_form(results, "Driver", before=202607)

    # The first race drops out for VER; LAW's missed rounds don't count against him.
    assert form["VER"] == 1
    assert form["LAW"] == 5
    assert predictions.recent_form(results, "Driver", before=202601).empty


def test_scores_compare_an_order_with_the_result():
    race = pd.DataFrame(
        {
            "Year": 2026,
            "Round": 1,
            "Driver": ["NOR", "PIA", "VER", "RUS"],
            "Predicted": [1, 2, 3, 4],
            "QualiPosition": [4.0, 1.0, 2.0, 3.0],
            "Position": [1.0, 3.0, 4.0, 2.0],
        }
    )

    model = predictions.scores(race, "Predicted")
    grid = predictions.scores(race, "QualiPosition")

    assert model == {"winner": 1.0, "podium": round(2 / 3, 3), "position_error": 1.0}
    assert grid["winner"] == 0.0
    assert grid["position_error"] == 2.0


def test_every_race_gets_a_full_predicted_order():
    rng = np.random.default_rng(0)
    rows = []
    for round_number in range(1, 31):
        grid = rng.permutation(10) + 1
        rows.append(
            pd.DataFrame(
                {
                    "Year": 2026,
                    "Round": round_number,
                    "Driver": [f"D{n}" for n in range(10)],
                    "Team": "Team",
                    "QualiPosition": grid.astype(float),
                    "QualiGap": grid * 0.1,
                    "LongRun": grid * 0.2,
                    "Form": grid.astype(float),
                    "TeamForm": grid.astype(float),
                    # The last race hasn't been run yet.
                    "Position": grid.astype(float) if round_number < 30 else np.nan,
                }
            )
        )
    table = pd.concat(rows, ignore_index=True)

    predicted = race_model.predict_races(table, folds=3)

    for _, race in predicted.groupby("Round"):
        assert sorted(race["Predicted"]) == list(range(1, 11))
    upcoming = predicted[predicted["Round"] == 30]
    assert upcoming["Position"].isna().all()
    # Every leaf needs 40 results, so the front row shares one and can tie.
    assert upcoming.sort_values("Predicted")["QualiPosition"].iloc[0] <= 2
