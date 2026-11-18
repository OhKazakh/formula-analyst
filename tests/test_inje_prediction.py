import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "research" / "inje-amr25" / "predict.py"


@pytest.fixture(scope="module")
def predict():
    spec = importlib.util.spec_from_file_location("inje_predict", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_classes_bridge_the_circuits_on_a_straight_line(predict):
    a, b, r = predict.fit_bridge(predict.BRIDGE)

    assert a == pytest.approx(0.660, abs=0.001)
    assert b == pytest.approx(9.81, abs=0.01)
    assert r == pytest.approx(0.997, abs=0.001)
    assert predict.class_ratio(predict.BRIDGE) == pytest.approx(0.7296, abs=0.0001)


def test_the_era_factor_comes_from_suzuka_poles(predict):
    assert predict.era_factor() == pytest.approx(86.983 / 90.915)
    assert predict.era_factor() == pytest.approx(0.957, abs=0.001)


def test_the_2013_pole_bridges_to_the_expected_range(predict):
    fit = predict.fit_bridge(predict.BRIDGE)
    pole = predict.seconds(predict.F1_KIC_2013["qualifying"])

    result = predict.predict(pole, fit, predict.class_ratio(predict.BRIDGE), predict.era_factor())

    assert predict.lap_time(result["ratio"], 1) == "1:07.8"
    assert predict.lap_time(result["linear"], 1) == "1:10.8"


@pytest.mark.parametrize(
    ("value", "decimals", "text"),
    [(67.849, 3, "1:07.849"), (70.78, 1, "1:10.8"), (59.96, 1, "1:00.0"), (95.0, 0, "1:35")],
)
def test_lap_times_read_as_minutes_and_seconds(predict, value, decimals, text):
    assert predict.lap_time(value, decimals) == text
    assert predict.seconds("1:35.239") == pytest.approx(95.239)


def test_the_simulator_holds_a_circle_at_its_corner_speed(predict):
    angle = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.column_stack([100 * np.cos(angle), 100 * np.sin(angle)])
    _, points = predict.resample(np.vstack([circle, circle[:1]]))
    # 20 km/h times the square root of the radius: 200 km/h on a 100 m circle.
    corners = {"slope": 0.5, "intercept": math.log(20)}
    car = {
        "accel": (np.array([0.0, 100.0]), np.array([10.0, 10.0])),
        "brake": (np.array([0.0, 100.0]), np.array([30.0, 30.0])),
    }

    speed = predict.simulate(predict.radii(points), predict.STEP, corners, car)

    assert speed * 3.6 == pytest.approx(200, rel=0.02)
    assert predict.lap_seconds(speed, predict.STEP) == pytest.approx(
        2 * math.pi * 100 / (200 / 3.6), rel=0.02
    )
