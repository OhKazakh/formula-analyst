from __future__ import annotations

import re
from collections import Counter

import pandas as pd
import plotly.graph_objects as go

from src.charts import ACCENT, OUTLINE, finish

RAIN = "rgba(0, 103, 173, 0.18)"
MESSAGE_TYPES = ["Penalty", "Investigation", "Track limits", "Safety car", "Flag", "DRS", "Other"]
PENALTY_COLUMNS = ["Lap", "Driver", "Penalty", "Reason"]
PENALTY = re.compile(
    r"^(?!(?:FIA STEWARDS: )?PENALTY SERVED).*?"
    r"(?P<penalty>(?:\d+ SECOND (?:TIME|STOP/GO) |DRIVE THROUGH |STOP-AND-GO )?PENALTY"
    r"|REPRIMAND(?: \(\w+\))?) "
    r"FOR CAR \d+ \((?P<driver>[A-Z]{3})\)\s*(?:-\s*(?P<reason>[^(]+))?"
)
DELETED = re.compile(r"^CAR \d+ \((?P<driver>[A-Z]{3})\) (?:TIME [\d:.]+ |LAP )DELETED")
WARNED = re.compile(r"^BLACK AND WHITE FLAG (?:FOR )?CAR \d+ \((?P<driver>[A-Z]{3})\)")


def lap_end_times(laps: pd.DataFrame) -> pd.DataFrame:
    timed = laps.dropna(subset=["Time"])
    ends = timed.groupby("LapNumber")["Time"].min().dt.total_seconds()
    return pd.DataFrame({"LapNumber": ends.index.astype(int), "Seconds": ends.to_numpy()})


def weather_by_lap(weather: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    columns = ["LapNumber", "AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Rainfall"]
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


# Laps line up across drivers only in a race, so other sessions are plotted against time.
def weather_by_minute(weather: pd.DataFrame) -> pd.DataFrame:
    columns = ["Minute", "AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Rainfall"]
    readings = weather.dropna(subset=["Time", "TrackTemp"]).sort_values("Time")
    if readings.empty:
        return pd.DataFrame(columns=columns)
    minutes = (readings["Time"] // 60).astype(int)
    return readings.assign(Minute=minutes)[columns].reset_index(drop=True)


def weather_summary(frame: pd.DataFrame, x: str = "LapNumber") -> dict[str, float | list[int]]:
    if frame.empty:
        return {}
    rain = frame.loc[frame["Rainfall"].astype(bool), x].astype(int).tolist()
    return {
        "track_start": float(frame["TrackTemp"].iloc[0]),
        "track_end": float(frame["TrackTemp"].iloc[-1]),
        "air_start": float(frame["AirTemp"].iloc[0]),
        "humidity": float(frame["Humidity"].median()),
        "wind": float(frame["WindSpeed"].max()),
        "rain": rain,
    }


def weather_figure(frame: pd.DataFrame, x: str = "LapNumber") -> go.Figure:
    label = "Lap" if x == "LapNumber" else "Minute"
    figure = go.Figure()
    for value in frame.loc[frame["Rainfall"].astype(bool), x]:
        figure.add_vrect(
            x0=value - 0.5, x1=value + 0.5, fillcolor=RAIN, line_width=0, layer="below"
        )
    figure.add_trace(
        go.Scatter(
            x=frame[x],
            y=frame["TrackTemp"],
            name="Track",
            mode="lines",
            line={"color": ACCENT, "width": 2.5},
            hovertemplate=f"{label} %{{x}}<br>track %{{y:.1f}} °C<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame[x],
            y=frame["AirTemp"],
            name="Air",
            mode="lines",
            line={"color": OUTLINE, "width": 2, "dash": "dash"},
            hovertemplate=f"{label} %{{x}}<br>air %{{y:.1f}} °C<extra></extra>",
        )
    )
    figure.update_xaxes(title=label if x == "LapNumber" else "Minutes into the session")
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


def _penalty_name(text: str) -> str:
    name = text.lower().replace(" second ", " s ").replace("stop/go", "stop-go")
    name = name.replace("stop-and-go", "stop-go").replace("drive through", "drive-through")
    return re.sub(r" \(\w+\)", "", name).capitalize()


def penalties(messages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lap, message in zip(messages["Lap"], messages["Message"], strict=True):
        if match := PENALTY.match(message.strip()):
            rows.append(
                {
                    "Lap": lap,
                    "Driver": match["driver"],
                    "Penalty": _penalty_name(match["penalty"]),
                    "Reason": (match["reason"] or "").strip().capitalize(),
                }
            )
    return pd.DataFrame(rows, columns=PENALTY_COLUMNS)


def track_limits(messages: pd.DataFrame) -> pd.DataFrame:
    deleted, warned = Counter(), set()
    for message in messages["Message"]:
        if match := DELETED.match(message.strip()):
            deleted[match["driver"]] += 1
        elif match := WARNED.match(message.strip()):
            warned.add(match["driver"])
    drivers = sorted(set(deleted) | warned, key=lambda driver: (-deleted[driver], driver))
    return pd.DataFrame(
        {
            "Driver": drivers,
            "Deleted": [deleted[driver] for driver in drivers],
            "Warned": [driver in warned for driver in drivers],
        }
    )


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
