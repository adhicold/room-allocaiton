"""Streamlit front-end: room finder, admin dashboard and evaluation viewer.

Run with ``streamlit run app/streamlit_app.py``. All logic lives in the
``roomfinder`` package; this file only handles presentation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from roomfinder.config import OPEN_END, OPEN_START, Weights
from roomfinder.data import load_campus
from roomfinder.evaluate import run, term_bounds
from roomfinder.matcher import Request, recommend, suggest_alternatives
from roomfinder.occupancy import build_occupancy, class_clashes, weekly_grid
from roomfinder.utilization import campus_heatmap, plan_moves, room_utilisation

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, REPORT_DIR = ROOT / "data" / "raw", ROOT / "reports"

st.set_page_config(page_title="RoomFinder", page_icon="🏫", layout="wide")


@st.cache_resource(show_spinner="Loading campus data...")
def get_campus():
    campus = load_campus(DATA_DIR)
    lo, hi = term_bounds(campus)
    occ = build_occupancy(campus.rooms, campus.schedule, campus.events, lo, hi)
    return campus, occ, lo, hi


@st.cache_data(show_spinner="Analysing timetable...")
def get_admin_tables():
    campus, _, _, _ = get_campus()
    grid = weekly_grid(campus.rooms, campus.schedule)
    moves, summary = plan_moves(campus.rooms, campus.schedule)
    util = room_utilisation(campus.rooms, grid)
    return grid, util, class_clashes(campus.schedule), moves, summary


def find_room_page() -> None:
    campus, base_occ, lo, hi = get_campus()
    if "occ" not in st.session_state:
        st.session_state.occ = base_occ.copy()  # live bookings made in this session
    occ = st.session_state.occ

    st.header("Find a room")
    with st.sidebar:
        st.subheader("Request")
        date = st.date_input("Date", value=pd.Timestamp("2025-09-11"),
                             min_value=lo.date(), max_value=hi.date())
        start = st.slider("Start hour", OPEN_START, OPEN_END, 14)
        duration = st.slider("Duration (hours)", 1, 4, 2)
        capacity = st.number_input("Attendees", 1, 200, 40)
        projector = st.checkbox("Projector")
        computers = st.checkbox("Computers")
        accessible = st.checkbox("Accessible room")
        buildings = ["Any", *sorted(campus.rooms["building"].unique())]
        building = st.selectbox("Preferred building", buildings)
        with st.expander("Ranking weights"):
            w = Weights(
                st.slider("Capacity fit", 0.0, 1.0, 0.55),
                st.slider("Avoid wasting equipment", 0.0, 1.0, 0.25),
                st.slider("Building preference", 0.0, 1.0, 0.10),
                st.slider("Accessible bonus", 0.0, 1.0, 0.10),
            )
        if st.button("Reset bookings"):
            st.session_state.occ = base_occ.copy()
            st.rerun()

    req = Request(pd.Timestamp(date), start, duration, int(capacity), projector, computers,
                  accessible, None if building == "Any" else building)
    try:
        recs = recommend(campus.rooms, occ, req, w)
    except ValueError as exc:
        st.error(str(exc))
        return

    if not recs:
        st.warning("No room satisfies every constraint. Nearest alternative start times:")
        for alt in suggest_alternatives(campus.rooms, occ, req, w):
            r = alt["room"]
            st.write(f"**{alt['start_hour']:02d}:00** ({alt['shift_hours']:+d} h) - "
                     f"{r.room_id}, {r.building}, {r.capacity} seats")
        return

    for i, r in enumerate(recs, 1):
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f"**{i}. {r.room_id}** - {r.building}, {r.capacity} seats "
                          f"(score {r.score:.2f})")
            left.markdown("\n".join(f"- {x}" for x in r.reasons))
            if right.button("Book", key=f"book_{r.room_id}"):
                occ.book(r.room_id, occ.slot(req.date, req.start_hour), req.duration_hours)
                st.success(f"{r.room_id} booked. It will not be offered for this slot again.")
                st.rerun()

    day0 = occ.slot(req.date, 0)
    rows = [occ.room_row(r.room_id) for r in recs]
    view = (occ.grid[rows, day0 + OPEN_START : day0 + OPEN_END + 1] > 0).astype(int)
    fig = px.imshow(view, x=list(range(OPEN_START, OPEN_END + 1)), y=[r.room_id for r in recs],
                    color_continuous_scale=["#d9f2e3", "#e8837c"], aspect="auto",
                    labels={"x": "Hour", "color": "Busy"}, title="Day view (red = busy)")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def admin_page() -> None:
    st.header("Admin dashboard")
    campus, *_ = get_campus()
    grid, util, clashes, moves, summary = get_admin_tables()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overlapping class pairs", f"{len(clashes):,}")
    def delta(key: str) -> str:
        return f"{summary['after'][key] - summary['before'][key]:,} after repair"

    c2.metric("Double-booked slots", f"{summary['before']['double_booked_slots']:,}",
              delta("double_booked_slots"), delta_color="inverse")
    c3.metric("Over-crowded classes", f"{summary['before']['overcrowded_classes']:,}",
              delta("overcrowded_classes"), delta_color="inverse")
    c4.metric("Underused rooms (<25 %)", int(util["underused"].sum()))

    heat, rooms_tab, clash_tab, move_tab = st.tabs(
        ["Utilisation heatmap", "Rooms", "Clash report", "Proposed moves"])
    with heat:
        hm = campus_heatmap(grid)
        st.plotly_chart(px.imshow(hm, aspect="auto", color_continuous_scale="YlOrRd",
                                  labels={"x": "Hour", "y": "Weekday", "color": "Occupied"}),
                        use_container_width=True)
    with rooms_tab:
        st.dataframe(util.sort_values("utilisation"), use_container_width=True, hide_index=True)
    with clash_tab:
        st.dataframe(clashes, use_container_width=True, hide_index=True)
        st.download_button("Download clashes CSV", clashes.to_csv(index=False),
                           "class_clashes.csv")
    with move_tab:
        st.caption("Greedy repair: a move never creates a new clash and always fits the class.")
        st.dataframe(moves, use_container_width=True, hide_index=True)
        st.download_button("Download moves CSV", moves.to_csv(index=False), "proposed_moves.csv")


def evaluation_page() -> None:
    st.header("Model evaluation")
    path = REPORT_DIR / "metrics.json"
    if not path.exists():
        with st.spinner("Running evaluation (about 15 s)..."):
            run(DATA_DIR, REPORT_DIR)
    metrics = json.loads(path.read_text())
    st.subheader("Allocation policies (replay of all non-cancelled requests)")
    st.dataframe(pd.DataFrame(metrics["allocation_policies"]), hide_index=True)
    st.subheader("Learned models vs baselines")
    st.json({"demand_forecast": metrics["demand_forecast"],
             "unmet_request_classifier": metrics["unmet_request_classifier"]})
    st.markdown((REPORT_DIR / "REPORT.md").read_text())


PAGES = {"Find a room": find_room_page, "Admin dashboard": admin_page,
         "Evaluation": evaluation_page}
choice = st.sidebar.radio("Page", list(PAGES))
PAGES[choice]()
