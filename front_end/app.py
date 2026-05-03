"""
Volatility Modelling Dashboard
Three tabs (top navigation):
  1. Model Specification  – model details, metrics, and data pipeline
  2. Individual Stock     – volatility deep-dive with stock selector
  3. Universe             – correlation-based grouping / cluster map
"""

import sys
import os

# Ensure the front_end directory is on the path so page modules can import
# charts, stock_registry, etc. as siblings.
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

st.set_page_config(
    page_title="Volatility Dashboard",
    page_icon="📈",
    layout="wide"
)

from pages.model_spec import render as render_model_spec
from pages.individual import render as render_individual
from pages.universe import render as render_universe

# Fixed top nav bar — stays visible regardless of scroll position.
st.markdown(
    """
    <style>
    /* Remove Streamlit's own header so our bar sits at the true top */
    header[data-testid="stHeader"] {
        display: none;
    }

    /* Fixed, full-width nav bar */
    [data-baseweb="tab-list"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 9999;
        background-color: var(--background-color, #ffffff) !important;
        background: var(--background-color, #ffffff) !important;
        opacity: 1 !important;
        isolation: isolate;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
        padding: 0 2.5rem;
        height: 3.25rem;
    }

    @media (prefers-color-scheme: dark) {
        [data-baseweb="tab-list"] {
            background-color: #0e1117 !important;
            background: #0e1117 !important;
        }
    }

    [data-baseweb="tab"] {
        height: 3.25rem;
        font-size: 0.95rem;
        font-weight: 500;
        padding: 0 1.25rem;
    }

    /* Push page content below the fixed bar */
    .stMainBlockContainer {
        padding-top: 4.5rem !important;
    }
    """,
    unsafe_allow_html=True,
)

tab_model, tab_individual, tab_universe = st.tabs(
    ["📋  Model Specification", "📈  Individual Stock", "🌐  Universe"]
)

with tab_model:
    render_model_spec()

with tab_individual:
    render_individual()

with tab_universe:
    render_universe()
