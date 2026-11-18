"""The loading state: start lights coming on one by one, with something to read meanwhile."""

from __future__ import annotations

import html
import random
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from views.theme import SECONDARY_TEXT

# Like st.spinner: a load that's done sooner never shows the lights, so cached reruns don't flash.
DELAY = 0.5
LIGHTS = 5
CYCLE = 4.0  # seconds: a light every 0.6 s, then lights out
LIGHT_ON = "#E10600"
LIGHT_OFF = "#47464C"
GANTRY = "light-dark(#15151E, #303037)"
TIP_SHARE = 0.2

TIPS = [
    "Tip: pick a favourite driver in the sidebar to pick them out in every chart and table.",
    "Tip: the Replay tab plays the race at up to 4x speed, or in real time.",
    "Tip: the Predictions page checks every race forecast against the result.",
    "Tip: the Teammates view on the Championship page shows who beat whom.",
]


def _keyframes(light: int) -> str:
    on = (light + 1) * 0.6 / CYCLE * 100
    out = 3.6 / CYCLE * 100
    lit = f"background: {LIGHT_ON}; box-shadow: 0 0 10px rgba(225, 6, 0, 0.7);"
    dark = f"background: {LIGHT_OFF}; box-shadow: none;"
    return (
        f"@keyframes start-light-{light} {{ 0%, {on - 0.01:.2f}% {{ {dark} }} "
        f"{on:.2f}%, {out:.2f}% {{ {lit} }} {out + 0.01:.2f}%, 100% {{ {dark} }} }}"
        f".start-lights span:nth-child({light + 1}) {{ animation: start-light-{light} "
        f"{CYCLE}s infinite; }}"
    )


STYLE = f"""<style>
.start-loader {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1rem;
  padding: 1rem 0; animation: start-loader-in 0.2s ease-out; }}
.start-lights {{ display: flex; gap: 0.45rem; padding: 0.5rem 0.65rem; border-radius: 999px;
  background: {GANTRY}; }}
.start-lights span {{ width: 1rem; height: 1rem; border-radius: 50%; background: {LIGHT_OFF}; }}
.start-loader-text {{ display: flex; flex-direction: column; gap: 0.15rem; min-width: 14rem;
  flex: 1; }}
.start-loader-text strong {{ font-weight: 600; }}
.start-loader-text span {{ color: {SECONDARY_TEXT}; font-size: 0.9rem; }}
@keyframes start-loader-in {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
{"".join(_keyframes(light) for light in range(LIGHTS))}
@media (prefers-reduced-motion: reduce) {{
  .start-loader {{ animation: none; }}
  .start-lights span {{ animation: none !important; background: {LIGHT_ON}; }}
}}
</style>"""


def pick(facts: list[str]) -> str:
    if not facts or random.random() < TIP_SHARE:
        return random.choice(TIPS)
    return random.choice(facts)


def loader_html(label: str, note: str) -> str:
    lights = "<span></span>" * LIGHTS
    return (
        f'{STYLE}<div class="start-loader" role="status" aria-live="polite">'
        f'<div class="start-lights" aria-hidden="true">{lights}</div>'
        f'<div class="start-loader-text"><strong>{html.escape(label)}</strong>'
        f"<span>{html.escape(note)}</span></div></div>"
    )


@contextmanager
def loading(label: str, facts: Callable[[], list[str]] | None = None) -> Iterator[None]:
    """Show the start lights and a fact while the block runs, if it takes a while.

    `facts` is only called once the lights are due, so a quick load never pays for it.
    """
    slot = st.empty()
    lock = threading.Lock()
    running = True

    def show() -> None:
        note = pick(facts() if facts else [])
        with lock:
            if running:
                slot.html(loader_html(label, note))

    timer = threading.Timer(DELAY, show)
    add_script_run_ctx(timer)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        with lock:
            running = False
        slot.empty()
