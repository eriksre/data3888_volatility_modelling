import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from stock_registry import ALL_BOOK_STEMS


MODEL_OPTIONS = ["Transformer", "LSTM", "GRU", "XGBoost", "HAR-RV", "GARCH(1,1)"]


@st.cache_data
def _load_universe_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    stocks = ALL_BOOK_STEMS

    summary_df = pd.DataFrame({
        "stock_id": stocks,
        "mean_volatility": rng.uniform(0.0015, 0.0050, len(stocks)),
        "rmse": rng.uniform(0.00015, 0.00060, len(stocks)),
        "qlike": rng.uniform(0.010, 0.030, len(stocks)),
        "best_model": rng.choice(MODEL_OPTIONS, len(stocks)),
    })

    factors = rng.normal(size=(len(stocks), 4))
    corr = np.corrcoef(factors @ factors.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    corr_df = pd.DataFrame(corr, index=stocks, columns=stocks)

    return summary_df, corr_df


def _ranking_chart(top_df: pd.DataFrame, metric_col: str, ranking_metric: str):
    fig = px.bar(
        top_df,
        x="stock_id",
        y=metric_col,
        color="best_model",
        labels={
            "stock_id": "Stock",
            metric_col: ranking_metric,
            "best_model": "Best model",
        },
        color_discrete_sequence=[
            "steelblue",
            "darkorange",
            "#6b7280",
            "#2f9e44",
            "#845ef7",
            "#e03131",
        ],
    )
    fig.update_layout(
        height=460,
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=70, b=40, l=20, r=20),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    return fig


def _correlation_heatmap(corr_df: pd.DataFrame, stocks: list[str]):
    fig = px.imshow(
        corr_df.loc[stocks, stocks],
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels=dict(color="Corr"),
    )
    fig.update_layout(
        height=620,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=30, l=20, r=20),
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def render() -> None:
    st.title("Stock Universe")

    summary_df, corr_df = _load_universe_data()

    st.caption("Cross-stock overview for volatility behaviour and model performance.")

    st.subheader("Universe Controls")
    c1, c2, c3 = st.columns([1.2, 1.2, 1])

    with c1:
        ranking_metric = st.selectbox(
            "Ranking metric",
            ["Mean Volatility", "RMSE", "QLIKE"],
            key="universe_ranking_metric",
        )

    with c2:
        top_n = st.slider(
            "Top-N stocks to display",
            min_value=5,
            max_value=30,
            value=10,
            key="universe_top_n",
        )

    with c3:
        sort_order = st.radio(
            "Sort order",
            ["Descending", "Ascending"],
            horizontal=True,
            key="universe_sort_order",
        )

    metric_map = {
        "Mean Volatility": "mean_volatility",
        "RMSE": "rmse",
        "QLIKE": "qlike",
    }
    metric_col = metric_map[ranking_metric]
    ascending = sort_order == "Ascending"

    ranked_df = summary_df.sort_values(by=metric_col, ascending=ascending).reset_index(drop=True)
    top_df = ranked_df.head(top_n)

    num_stocks = len(summary_df)
    avg_vol = summary_df["mean_volatility"].mean()
    most_volatile_stock = summary_df.loc[summary_df["mean_volatility"].idxmax(), "stock_id"]
    hardest_stock = summary_df.loc[summary_df["rmse"].idxmax(), "stock_id"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks", f"{num_stocks}")
    m2.metric("Average volatility", f"{avg_vol:.6f}")
    m3.metric("Most volatile", most_volatile_stock)
    m4.metric("Hardest to predict", hardest_stock)

    st.divider()

    st.subheader(f"Top {top_n} Stocks by {ranking_metric}")
    st.plotly_chart(
        _ranking_chart(top_df, metric_col, ranking_metric),
        width="stretch",
    )

    st.subheader("Stock Similarity")
    st.caption("Correlation view for the currently ranked stocks.")
    st.plotly_chart(
        _correlation_heatmap(corr_df, top_df["stock_id"].tolist()),
        width="stretch",
    )

    left, right = st.columns([2.2, 1])

    with left:
        st.subheader("Per-Stock Summary")
        st.dataframe(
            ranked_df.rename(columns={
                "stock_id": "Stock",
                "mean_volatility": "Mean volatility",
                "rmse": "RMSE",
                "qlike": "QLIKE",
                "best_model": "Best model",
            }),
            hide_index=True,
            width="stretch",
            column_config={
                "Mean volatility": st.column_config.NumberColumn(format="%.6f"),
                "RMSE": st.column_config.NumberColumn(format="%.6f"),
                "QLIKE": st.column_config.NumberColumn(format="%.5f"),
            },
        )

    with right:
        st.subheader("Open Stock")
        selected_stock = st.selectbox(
            "Select a stock",
            options=ranked_df["stock_id"].tolist(),
            key="universe_selected_stock",
        )

        if st.button("Open in Individual Stock", type="primary", width="stretch"):
            st.session_state.selected_stock = selected_stock
            st.success(f"{selected_stock} is selected. Open the Individual Stock tab to view it.")

        st.info(
            "Use this view to compare stocks globally, find high-volatility names, "
            "and identify instruments with similar behaviour."
        )
