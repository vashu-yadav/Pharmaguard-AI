"""Main Streamlit application for PharmaGuard AI."""

from pathlib import Path
import sys

import streamlit as st

# Make repository packages importable whether Streamlit is started from the
# repository root or from VS Code's app folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_FOLDER = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_FOLDER) not in sys.path:
    sys.path.insert(0, str(APP_FOLDER))

from explainability_page import render_explainability_page
from home_page import render_home_page
from machine_page import render_machine_page
from performance_page import render_performance_page
from tablet_page import render_tablet_page

st.set_page_config(
    page_title="PharmaGuard AI | Manufacturing Intelligence",
    page_icon="\U0001F48A",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #16324f; --teal: #087f8c; --mint: #e8f5f2; }
    .stApp { background: linear-gradient(180deg, #f7fbfc 0%, #f4f7fb 55%, #eef4f7 100%); }
    [data-testid="stSidebar"] { background: #102a43; }
    [data-testid="stSidebar"] * { color: #f4fbff !important; }
    [data-testid="stMetric"] { background: white; border: 1px solid #dce8ec;
      padding: 15px 18px; border-radius: 14px; box-shadow: 0 3px 12px #16324f0b; }
    .hero { padding: 1.2rem 1.6rem; border-radius: 18px;
      background: linear-gradient(120deg, #102a43, #087f8c); color: white; margin-bottom: 1rem; }
    .hero h1 { color: white; margin-bottom: .2rem; }
    .hero p { color: #e5f4f4; font-size: 1.05rem; margin-bottom: .2rem; }
    .module-card { background: white; padding: 1.2rem 1.4rem; border-radius: 16px;
      border: 1px solid #dce8ec; min-height: 170px; }
    .muted { color: #526777; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## PharmaGuard AI")
    st.caption("Explainable manufacturing intelligence")
    page = st.radio(
        "Navigate",
        ["Home Dashboard", "Smart Manufacturing", "Tablet Quality Control", "Explainable AI", "Model Performance"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Two independent ML modules. Their data and predictions are not linked.")

if page == "Home Dashboard":
    render_home_page()
elif page == "Smart Manufacturing":
    render_machine_page()
elif page == "Tablet Quality Control":
    render_tablet_page()
elif page == "Explainable AI":
    render_explainability_page()
else:
    render_performance_page()

st.markdown(
    "<div class='muted' style='text-align:center;padding:2rem 0 1rem'>"
    "PharmaGuard AI | Decision support prototype | Visual checks do not verify medicine safety</div>",
    unsafe_allow_html=True,
)
