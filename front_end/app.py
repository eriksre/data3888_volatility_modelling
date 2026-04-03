"""
Volatility Modelling Dashboard
Two screens:
  1. Stock Universe   – correlation-based grouping / cluster map
  2. Individual Stock – volatility deep-dive dashboard
"""

import streamlit as st

from stock_registry import ALL_BOOK_STEMS, ALL_NAMED_TICKERS

st.set_page_config(
    page_title="Volatility Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "screen" not in st.session_state:
    st.session_state.screen = "universe"
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None


def render_universe() -> None:
    with st.sidebar:
        st.header("Stock Universe")
        st.divider()

        dataset = st.radio("Dataset", ["Named stocks (10)", "Book stocks (112)"])
        stock_list = ALL_NAMED_TICKERS if dataset == "Named stocks (10)" else ALL_BOOK_STEMS

        st.divider()

        st.selectbox(
            "Group by",
            ["Correlation", "Sector", "Mean Volatility", "Volatility Std Dev"],
        )

        st.slider("Top-N most volatile to highlight", min_value=0, max_value=20, value=5)

        st.divider()

        chosen = st.selectbox("Open stock dashboard", stock_list)
        if st.button("View stock →", type="primary", use_container_width=True):
            st.session_state.selected_stock = chosen
            st.session_state.screen = "individual"
            st.rerun()

    st.title("Stock Universe")



def render_individual() -> None:
    stock_id = st.session_state.selected_stock

    with st.sidebar:
        st.header("Individual Stock")
        st.divider()

        if st.button("← Back to Universe", use_container_width=True):
            st.session_state.screen = "universe"
            st.rerun()

        all_stocks = ALL_NAMED_TICKERS + ALL_BOOK_STEMS
        idx = all_stocks.index(stock_id) if stock_id in all_stocks else 0
        new_stock = st.selectbox("Switch stock", all_stocks, index=idx)
        if st.button("Go", use_container_width=True):
            st.session_state.selected_stock = new_stock
            st.rerun()

        st.divider()
        st.slider("Rolling window (periods)", 3, 20, 7)
        st.toggle("Show model forecast", value=True)

    st.title(f"Stock {stock_id}")
    st.info("add charts here")


def main() -> None:
    if st.session_state.screen == "universe":
        render_universe()
    else:
        render_individual()


main()


