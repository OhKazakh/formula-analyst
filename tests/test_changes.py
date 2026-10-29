import json

from src import changes


def save(folder, event, session=None):
    folder.mkdir(parents=True)
    meta = {"year": 2026, "event": event} | ({"session": session} if session else {})
    path = folder / "race.json"
    path.write_text(json.dumps(meta))
    return path


def test_commit_message_names_new_weekends(tmp_path):
    race = save(tmp_path / "19-mexico-city-grand-prix", "Mexico City Grand Prix")
    qualifying = save(
        tmp_path / "19-mexico-city-grand-prix" / "qualifying",
        "Mexico City Grand Prix",
        "Qualifying",
    )
    next_weekend = save(
        tmp_path / "20-sao-paulo-grand-prix" / "sprint", "São Paulo Grand Prix", "Sprint"
    )

    assert changes.describe([race, qualifying]) == "Add the 2026 Mexico City Grand Prix"
    assert (
        changes.describe([race, next_weekend])
        == "Add the 2026 Mexico City Grand Prix and the 2026 São Paulo Grand Prix sessions so far"
    )


def test_commit_message_without_new_sessions(tmp_path):
    assert (
        changes.describe([tmp_path / "results.parquet"], year=2026)
        == "Update the 2026 championship"
    )
