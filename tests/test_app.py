"""Smoke test: the Streamlit pages render without exceptions."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data/raw/rooms.csv").exists(), reason="no dataset")


def test_find_room_page_renders_results():
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.button) >= 2  # Reset + at least one Book button


def test_admin_page_renders():
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Admin dashboard").run()
    assert not at.exception
    assert len(at.metric) == 4
