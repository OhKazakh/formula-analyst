import streamlit as st


def requested(name: str) -> str:
    key = f"requested:{name}"
    if key not in st.session_state:
        st.session_state[key] = st.query_params.get(name, "")
    return st.session_state[key]


def requested_int(name: str) -> int | None:
    value = requested(name)
    return int(value) if value.isdigit() else None
