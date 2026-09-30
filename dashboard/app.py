"""ClimateTwin interactive dashboard — Streamlit entrypoint.

Run with:

    streamlit run dashboard/app.py

The app is a read-only view over already-trained artifacts. It loads the fitted
scaler, the trained ClimateLSTM checkpoint, the 2024 test split, and the
evaluation metrics once (cached), then drives ``src.scenario`` and ``src.impact``
to render three pages: an overview, an interactive forecast, and a what-if
scenario simulator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the project importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.core import (  # noqa: E402
    artifacts_available,
    available_cities,
    load_artifacts,
)
from dashboard.pages_impl import forecast, map_view, overview, scenario  # noqa: E402
from src.impact import IMPACT_DISCLAIMER  # noqa: E402
from src.scenario import SCENARIO_DISCLAIMER  # noqa: E402

# Pages that operate on a single selected city's artifacts.
PER_CITY_PAGES = {
    "Overview": overview.render,
    "Forecast & current state": forecast.render,
    "What-if scenario simulator": scenario.render,
}
# The multi-city map reads the whole DB, not one city's artifacts.
PAGE_ORDER = [
    "Overview",
    "Multi-city map",
    "Forecast & current state",
    "What-if scenario simulator",
]


def main() -> None:
    st.set_page_config(
        page_title="ClimateTwin",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🌍 ClimateTwin")
    st.sidebar.caption("Localized AI climate digital twin")

    ok, missing = artifacts_available()
    if not ok:
        st.title("ClimateTwin dashboard")
        st.error(
            "Required trained artifacts are missing, so the dashboard cannot "
            "start. The dashboard is a view over existing artifacts and does "
            "not train anything itself."
        )
        st.markdown("**Missing files:**")
        for m in missing:
            st.markdown(f"- `{m}`")
        st.info(
            "Run the data pipeline and training to produce these, then reload. "
            "See `docs/CHECKPOINT.md` for the reproduction steps."
        )
        return

    page_name = st.sidebar.radio("Page", PAGE_ORDER)
    st.sidebar.divider()

    st.title("ClimateTwin")

    if page_name == "Multi-city map":
        # The map spans all cities; it reads the DB directly. We still pass the
        # primary city's artifacts so shared helpers (feature labels) work.
        map_view.render(load_artifacts())
    else:
        cities = available_cities()
        labels = {c.name: c.slug for c in cities}
        chosen_name = st.sidebar.selectbox(
            "City",
            list(labels.keys()),
            help="Choose which city's trained twin to explore.",
        )
        art = load_artifacts(labels[chosen_name])
        st.sidebar.caption(
            f"Location: **{art.location}**  \n"
            f"Model: **LSTM** ({art.metrics.get('parameter_count', 0):,} params)  \n"
            f"Test period: **2024** ({len(art.test_scaled):,} hours)"
        )
        PER_CITY_PAGES[page_name](art)

    st.divider()
    st.caption(f"⚠️ {SCENARIO_DISCLAIMER}")
    st.caption(f"ℹ️ {IMPACT_DISCLAIMER}")


if __name__ == "__main__":
    main()
