import pandas as pd
import streamlit as st

from src import races
from views import data, logos, theme
from views.state import favourite
from views.text import plural

BADGE = st.column_config.ImageColumn(" ", width=44)
SMALL = st.column_config.NumberColumn(width=80)


def weekend_predictions(race: races.Race, bundle_version: str) -> pd.DataFrame:
    table = data.race_predictions(bundle_version)
    return table[(table["Year"] == race.year) & (table["Round"] == race.round_number)]


def badges(race: races.Race, table: pd.DataFrame) -> list[str]:
    return [
        logos.badge(str(team), race.driver_styles.get(driver, {}).get("color"), race.year)
        for driver, team in zip(table["Driver"], table["Team"], strict=True)
    ]


def model_record(bundle_version: str) -> str:
    metrics = data.prediction_metrics(bundle_version)
    if not metrics:
        return ""
    model, grid = metrics["model"], metrics["qualifying"]
    return (
        f"Over {metrics['races']} races it picked the winner {model['winner']:.0%} of the time, "
        f"against {grid['winner']:.0%} for pole position, and was {model['position_error']:.2f} "
        f"places off on average, against {grid['position_error']:.2f} for the qualifying order: "
        "qualifying decides most races."
    )


def how_it_works(bundle_version: str) -> str:
    return (
        "The model reads each driver's qualifying position and gap to pole, best practice "
        "long run and average finish over the last five races, for the driver and the team. "
        "Every race is predicted by a model trained on the other races. "
        + model_record(bundle_version)
    )


def prediction_tab(race: races.Race, bundle_version: str) -> None:
    st.subheader("How predictable was it?")
    weekend = weekend_predictions(race, bundle_version).dropna(subset=["Position"])
    if weekend.empty:
        st.info("There's no prediction for this race: it needs its qualifying session.")
        return
    weekend = weekend.sort_values("Position")
    picked = weekend.loc[weekend["Predicted"].idxmin()]
    winner = weekend.iloc[0]
    if picked["Driver"] == winner["Driver"]:
        text = (
            f"The model picked **{winner['Driver']}** to win from P{winner['QualiPosition']:.0f} "
            "in qualifying, and called it."
        )
    else:
        text = (
            f"The model picked **{picked['Driver']}** (P{picked['QualiPosition']:.0f} in "
            f"qualifying) to win. **{winner['Driver']}** won from "
            f"P{winner['QualiPosition']:.0f}."
        )
    podium = set(weekend.loc[weekend["Predicted"] <= 3, "Driver"]) & set(
        weekend.loc[weekend["Position"] <= 3, "Driver"]
    )
    model_error = (weekend["Predicted"] - weekend["Position"]).abs().mean()
    grid_error = (weekend["QualiPosition"] - weekend["Position"]).abs().mean()
    text += (
        f" It had {len(podium)} of the podium in its top three and was {model_error:.1f} places "
        f"off on average, against {grid_error:.1f} for the qualifying order."
    )
    st.markdown(text)

    beat = weekend["Predicted"] - weekend["Position"]
    table = pd.DataFrame(
        {
            "Finished": weekend["Position"].astype(int).to_numpy(),
            "Badge": badges(race, weekend),
            "Driver": weekend["Driver"].to_numpy(),
            "Predicted": weekend["Predicted"].astype(int).to_numpy(),
            "Qualified": weekend["QualiPosition"].astype("Int64").to_numpy(),
            "Against the prediction": [
                "as predicted"
                if not places
                else f"{plural(int(abs(places)), 'place')} {'better' if places > 0 else 'worse'}"
                for places in beat
            ],
        }
    )
    st.dataframe(
        theme.favourite_rows(table, table["Driver"].eq(favourite())),
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,
        column_config={"Badge": BADGE, "Finished": SMALL, "Predicted": SMALL, "Qualified": SMALL},
    )
    st.caption(how_it_works(bundle_version))


def upcoming_prediction(race: races.Race, bundle_version: str) -> None:
    """The predicted race result, shown with qualifying until the race is run."""
    weekend = weekend_predictions(race, bundle_version)
    if weekend.empty or weekend["Position"].notna().any():
        return
    st.subheader("How will the race go?")
    weekend = weekend.sort_values("Predicted")
    favourite_pick = weekend.iloc[0]
    st.markdown(
        f"The model picks **{favourite_pick['Driver']}** to win from "
        f"P{favourite_pick['QualiPosition']:.0f}."
    )
    table = pd.DataFrame(
        {
            "Predicted": weekend["Predicted"].astype(int).to_numpy(),
            "Badge": badges(race, weekend),
            "Driver": weekend["Driver"].to_numpy(),
            "Qualified": weekend["QualiPosition"].astype("Int64").to_numpy(),
        }
    )
    st.dataframe(
        theme.favourite_rows(table, table["Driver"].eq(favourite())),
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,
        column_config={"Badge": BADGE, "Predicted": SMALL, "Qualified": SMALL},
    )
    st.caption(how_it_works(bundle_version))
