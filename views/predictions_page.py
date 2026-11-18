import pandas as pd
import streamlit as st

from src import predictions, races, seasons
from views import data, logos, theme
from views.links import requested_int
from views.prediction_tab import Weekend, forecast, next_race, review
from views.state import initial, pick_favourite, remember, remembered_round, remembered_season

RESULT_WIDTH = 84
# formula1.com's positive green and red, as on the race page.
RESULTS = {True: ("Called it", "#28973E", "#15151E"), False: ("Missed", "#E10600", "#FFFFFF")}


def points(share: float) -> str:
    """A difference between two shares in percentage points, as in "+3 points"."""
    count = round(share * 100)
    return f"{count:+d} point{'' if abs(count) == 1 else 's'}"


def overall(bundle_version: str) -> None:
    metrics = data.prediction_metrics(bundle_version)
    if not metrics:
        return
    model, grid = metrics["model"], metrics["qualifying"]
    winners, podium, places = st.columns(3)
    winners.metric(
        "Winners called",
        f"{model['winner']:.0%}",
        f"{points(model['winner'] - grid['winner'])} on pole position",
        help="How often the model's pick won, against how often the pole sitter won.",
    )
    podium.metric(
        "Podium places",
        f"{model['podium']:.0%}",
        f"{points(model['podium'] - grid['podium'])} on the qualifying order",
        help="How much of the podium was in the model's top three.",
    )
    places.metric(
        "Places off on average",
        f"{model['position_error']:.2f}",
        f"{model['position_error'] - grid['position_error']:+.2f} on the qualifying order",
        delta_color="inverse",
        help="How far each finish was from the predicted place.",
    )
    st.caption(
        f"Over {metrics['races']} races since 2018, each predicted by a model that never saw it."
    )


def record_table(record: pd.DataFrame, season: seasons.Season, table: pd.DataFrame) -> pd.DataFrame:
    teams = table[table["Year"] == season.year].set_index(["Round", "Driver"])["Team"]

    def badge(round_number: int, driver: str) -> str:
        color = season.driver_styles.get(driver, {}).get("color")
        return logos.badge(str(teams.get((round_number, driver), "")), color, season.year)

    return pd.DataFrame(
        {
            "Round": record["Round"],
            "Grand Prix": [
                season.event_name(r).removesuffix(" Grand Prix") for r in record["Round"]
            ],
            "Pick badge": [
                badge(r, d) for r, d in zip(record["Round"], record["Picked"], strict=True)
            ],
            "Picked": [
                f"{driver} from P{grid:.0f}"
                for driver, grid in zip(record["Picked"], record["PickedFrom"], strict=True)
            ],
            "Winner badge": [
                badge(r, d) for r, d in zip(record["Round"], record["Winner"], strict=True)
            ],
            "Winner": [
                f"{driver} from P{grid:.0f}"
                for driver, grid in zip(record["Winner"], record["WinnerFrom"], strict=True)
            ],
            "Result": [
                theme.pill(*RESULTS[bool(called)], RESULT_WIDTH) for called in record["Called"]
            ],
            "Podium": [f"{count} of 3" for count in record["Podium"]],
            "Places off": record["ModelError"].round(1),
            "Grid places off": record["GridError"].round(1),
        }
    )


def season_record(season: seasons.Season, table: pd.DataFrame) -> None:
    record = predictions.season_record(table, season.year)
    st.subheader(f"How did it do in {season.year}?")
    if record.empty:
        st.info("No race of this season has a prediction yet.")
        return
    pole_wins = int((record["WinnerFrom"] == 1).sum())
    st.markdown(
        f"It called **{int(record['Called'].sum())} of {len(record)}** winners, against "
        f"{pole_wins} for the pole sitter, and was {record['ModelError'].mean():.2f} places off "
        f"on average, against {record['GridError'].mean():.2f} for the qualifying order."
    )
    st.dataframe(
        record_table(record, season, table),
        hide_index=True,
        width="content",
        height=35 * (len(record) + 1) + 3,
        column_config={
            "Round": st.column_config.NumberColumn(width=60),
            "Grand Prix": st.column_config.TextColumn(width=150),
            "Pick badge": st.column_config.ImageColumn(" ", width=44),
            "Picked": st.column_config.TextColumn(width=110),
            "Winner badge": st.column_config.ImageColumn(" ", width=44),
            "Winner": st.column_config.TextColumn(width=110),
            "Result": st.column_config.ImageColumn("Result", width=RESULT_WIDTH + 8),
            "Podium": st.column_config.TextColumn(
                width=70, help="Podium finishers it had in its top three"
            ),
            "Places off": st.column_config.NumberColumn(
                width=80, help="How far each finish was from the predicted place, on average"
            ),
            "Grid places off": st.column_config.NumberColumn(
                width=110, help="The same for the qualifying order"
            ),
        },
    )


def render() -> None:
    bundle_version = races.bundle_version()
    table = data.race_predictions(bundle_version)
    st.title("Race predictions")
    if table.empty:
        st.info("No predictions are bundled.")
        return
    st.markdown(
        "After qualifying, a model trained on every race since 2018 predicts the finishing order "
        "from the grid, practice long runs and recent form. The forecast goes up on Saturday "
        "night and is checked against the result after the race."
    )
    overall(bundle_version)

    upcoming = next_race(bundle_version)
    if upcoming is None:
        st.subheader("What's next?")
        st.info(
            "No race is waiting to be run. The next forecast appears after qualifying, when the "
            "Saturday night update runs."
        )
    else:
        weekend, name, _ = upcoming
        forecast(weekend, bundle_version, f"How will the {name} go?")

    years = sorted(table["Year"].unique().tolist(), reverse=True)
    bundled = data.bundled_seasons(bundle_version)
    years = [year for year in years if year in bundled]

    def choose_year() -> int:
        wanted = remembered_season() or requested_int("season")
        return years.index(wanted) if wanted in years else 0

    year = st.sidebar.selectbox(
        "Season", years, index=initial("predictions_season", choose_year), key="predictions_season"
    )
    season = data.season(year, bundle_version)
    pick_favourite(sorted(season.results["Driver"].unique().tolist()))
    season_record(season, table)

    rounds = predictions.season_record(table, year)["Round"].tolist()
    if not rounds:
        return

    def choose_round() -> int:
        wanted = remembered_round(year) or requested_int("round")
        return rounds.index(wanted) if wanted in rounds else len(rounds) - 1

    key = f"predictions_round-{year}"
    chosen = st.selectbox(
        "Race",
        rounds,
        index=initial(key, choose_round),
        format_func=lambda r: f"Round {r} · {season.event_name(r)}",
        key=key,
    )
    remember(year, chosen)
    st.query_params.from_dict({"season": str(year), "round": str(chosen)})
    review(
        Weekend(year, chosen, season.driver_styles),
        bundle_version,
        f"How predictable was the {season.event_name(chosen)}?",
    )
