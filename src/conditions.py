from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.charts import ACCENT, OUTLINE, finish

RAIN = "rgba(0, 103, 173, 0.18)"
MESSAGE_TYPES = ["Penalty", "Investigation", "Track limits", "Safety car", "Flag", "DRS", "Other"]


def lap_end_times(laps: pd.DataFrame) -> pd.DataFrame:
    timed = laps.dropna(subset=["Time"])
    ends = timed.groupby("LapNumber")["Time"].min().dt.total_seconds()
    return pd.DataFrame({"LapNumber": ends.index.astype(int), "Seconds": ends.to_numpy()})


def weather_by_lap(weather: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    columns = ["LapNumber", "AirTemp", "TrackTemp", "Rainfall"]
    if weather.empty or "Time" not in laps:
        return pd.DataFrame(columns=columns)
    readings = weather.dropna(subset=["Time"]).sort_values("Time")
    merged = pd.merge_asof(
        lap_end_times(laps).sort_values("Seconds"),
        readings,
        left_on="Seconds",
        right_on="Time",
        direction="backward",
    )
    return merged[columns].dropna(subset=["TrackTemp"]).reset_index(drop=True)


def weather_summary(by_lap: pd.DataFrame) -> dict[str, float | list[int]]:
    if by_lap.empty:
        return {}
    rain = by_lap.loc[by_lap["Rainfall"].astype(bool), "LapNumber"].astype(int).tolist()
    return {
        "track_start": float(by_lap["TrackTemp"].iloc[0]),
        "track_end": float(by_lap["TrackTemp"].iloc[-1]),
        "air_start": float(by_lap["AirTemp"].iloc[0]),
        "rain_laps": rain,
    }


def weather_figure(by_lap: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for lap in by_lap.loc[by_lap["Rainfall"].astype(bool), "LapNumber"]:
        figure.add_vrect(x0=lap - 0.5, x1=lap + 0.5, fillcolor=RAIN, line_width=0, layer="below")
    figure.add_trace(
        go.Scatter(
            x=by_lap["LapNumber"],
            y=by_lap["TrackTemp"],
            name="Track",
            mode="lines",
            line={"color": ACCENT, "width": 2.5},
            hovertemplate="Lap %{x}<br>track %{y:.1f} °C<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=by_lap["LapNumber"],
            y=by_lap["AirTemp"],
            name="Air",
            mode="lines",
            line={"color": OUTLINE, "width": 2, "dash": "dash"},
            hovertemplate="Lap %{x}<br>air %{y:.1f} °C<extra></extra>",
        )
    )
    figure.update_xaxes(title="Lap")
    figure.update_yaxes(title="Temperature (°C)")
    return finish(figure, 320)


def message_type(category: str, message: str) -> str:
    text = message.upper()
    if any(word in text for word in ("INVESTIGATION", "NOTED", "REVIEWED", "NO FURTHER ACTION")):
        return "Investigation"
    if "PENALTY" in text:
        return "Penalty"
    if "TRACK LIMITS" in text or "DELETED" in text:
        return "Track limits"
    if category == "SafetyCar" or "SAFETY CAR" in text:
        return "Safety car"
    if category == "Flag":
        return "Flag"
    if category == "Drs":
        return "DRS"
    return "Other"


def race_control(messages: pd.DataFrame) -> pd.DataFrame:
    if messages.empty:
        return pd.DataFrame(columns=["Lap", "Type", "Message"])
    types = [
        message_type(category, message)
        for category, message in zip(messages["Category"], messages["Message"], strict=True)
    ]
    table = pd.DataFrame({"Lap": messages["Lap"], "Type": types, "Message": messages["Message"]})
    blue = table["Message"].str.contains("blue flag", case=False)
    return table[~blue].reset_index(drop=True)


def radio_by_lap(radio: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    columns = ["Lap", "Driver", "Url"]
    if radio.empty or "LapStartTime" not in laps:
        return pd.DataFrame(columns=columns)
    starts = laps.dropna(subset=["LapStartTime"]).assign(
        Seconds=lambda rows: rows["LapStartTime"].dt.total_seconds()
    )[["Driver", "LapNumber", "Seconds"]]
    clips = radio.dropna(subset=["Time"]).sort_values("Time")
    merged = pd.merge_asof(
        clips,
        starts.sort_values("Seconds"),
        left_on="Time",
        right_on="Seconds",
        by="Driver",
        direction="backward",
    )
    merged["Lap"] = merged["LapNumber"].fillna(0).astype(int)
    return merged[columns].reset_index(drop=True)
