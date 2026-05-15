import pandas as pd
import plotly.express as px
import streamlit as st

from back_end.config import normalize_stock_id
from back_end.service import get_latest_or_selected_run, load_universe_page_data

RANKING_METRICS = {
    "Mean Volatility": "mean_volatility",
    "MSE": "mse",
    "RMSE": "rmse",
    "MAE": "mae",
    "MAPE": "mape",
    "RMSPE": "rmspe",
    "QLIKE": "qlike",
}


def _stock_number(stock_id: str) -> int:
    text = str(stock_id)
    if text.startswith("stock_"):
        text = text.removeprefix("stock_")
    try:
        return int(text)
    except ValueError:
        return 10**9


def _stock_label(stock_id: str) -> str:
    text = str(stock_id)
    if text.startswith("stock_"):
        text = text.removeprefix("stock_")
    try:
        return f"Stock {int(text)}"
    except ValueError:
        return str(stock_id)


def _sort_by_stock_number(df: pd.DataFrame) -> pd.DataFrame:
    if "stock_id" not in df.columns:
        return df
    return (
        df.assign(_stock_number=df["stock_id"].map(_stock_number))
        .sort_values(["_stock_number", "stock_id"])
        .drop(columns="_stock_number")
        .reset_index(drop=True)
    )


def _with_stock_display(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    if "stock_id" in display_df.columns:
        display_df["stock_label"] = display_df["stock_id"].map(_stock_label)
    return display_df


def _stock_summary_view(summary_df: pd.DataFrame) -> pd.DataFrame:
    view = _with_stock_display(_sort_by_stock_number(summary_df)).drop(columns=["stock_id"])
    columns = ["stock_label", *[col for col in view.columns if col != "stock_label"]]
    return view[columns].rename(
        columns={
            "stock_label": "Stock",
            "mean_volatility": "Mean volatility",
            "mse": "MSE",
            "rmse": "RMSE",
            "mae": "MAE",
            "mape": "MAPE",
            "rmspe": "RMSPE",
            "qlike": "QLIKE",
            "best_model": "Best model",
        }
    )


def _parse_manual_stocks(raw_stocks: str, available_stocks: list[str]) -> tuple[list[str], list[str]]:
    available = set(available_stocks)
    selected: list[str] = []
    missing: list[str] = []

    for token in raw_stocks.replace(",", " ").split():
        stock = normalize_stock_id(token.strip())
        if stock in selected or stock in missing:
            continue
        if stock in available:
            selected.append(stock)
        else:
            missing.append(stock)

    return selected, missing


def _select_display_stocks(ranked_df: pd.DataFrame, manual_stocks: list[str], top_n: int) -> pd.DataFrame:
    selected_stocks = ranked_df.head(top_n)["stock_id"].tolist()
    selected_stocks.extend(stock for stock in manual_stocks if stock not in selected_stocks)
    return ranked_df.set_index("stock_id").loc[selected_stocks].reset_index()


def _load_universe_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[float], pd.DataFrame]:
    run_id = get_latest_or_selected_run(st.session_state.get("selected_run_id"))
    return load_universe_page_data(run_id)


def _ranking_chart(top_df: pd.DataFrame, metric_col: str, ranking_metric: str):
    plot_df = _with_stock_display(top_df)
    stock_order = plot_df["stock_label"].tolist()
    fig = px.bar(
        plot_df,
        x="stock_label",
        y=metric_col,
        color="best_model",
        category_orders={"stock_label": stock_order},
        labels={
            "stock_label": "Stock",
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
    plot_df = corr_df.loc[stocks, stocks].copy()
    labels = [_stock_label(stock) for stock in stocks]
    plot_df.index = labels
    plot_df.columns = labels
    fig = px.imshow(
        plot_df,
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


def _pca_axis_label(component: str, explained: list[float]) -> str:
    try:
        idx = int(component.replace("PC", "")) - 1
    except ValueError:
        return component
    if 0 <= idx < len(explained):
        return f"{component} ({explained[idx] * 100:.1f}% var.)"
    return component


def _pca_scatter(
    pca_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    x_component: str,
    y_component: str,
    explained: list[float],
):
    plot_df = _with_stock_display(pca_df.merge(summary_df, on="stock_id", how="left"))
    fig = px.scatter(
        plot_df,
        x=x_component,
        y=y_component,
        hover_name="stock_label",
        hover_data={
            "stock_id": False,
            "stock_label": False,
            "mean_volatility": ":.6f",
            "mse": ":.6f",
            "rmse": ":.6f",
            "mae": ":.6f",
            "mape": ":.3f",
            "rmspe": ":.6f",
            "qlike": ":.5f",
            x_component: ":.3f",
            y_component: ":.3f",
        },
        labels={
            x_component: _pca_axis_label(x_component, explained),
            y_component: _pca_axis_label(y_component, explained),
            "mean_volatility": "Mean volatility",
            "mse": "MSE",
            "rmse": "RMSE",
            "mae": "MAE",
            "mape": "MAPE",
            "rmspe": "RMSPE",
            "qlike": "QLIKE",
            "stock_label": "Stock",
        },
    )
    fig.update_layout(
        height=560,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=30, l=20, r=20),
        showlegend=False,
    )
    fig.update_traces(marker=dict(size=8, color="steelblue", line=dict(width=0.5, color="white")), selector=dict(mode="markers"))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)", zeroline=True, zerolinecolor="rgba(80,80,80,0.35)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)", zeroline=True, zerolinecolor="rgba(80,80,80,0.35)")
    return fig


def _model_comparison_view(model_df: pd.DataFrame, winner_metric: str) -> pd.DataFrame:
    best_col = f"best_stocks_{winner_metric}"
    metric_cols = [winner_metric, *[metric for metric in ["rmse", "qlike", "mae", "mse", "mape", "rmspe"] if metric != winner_metric]]
    display_cols = [
        "model",
        "model_type",
        "mean_inference_ms",
        best_col,
        *metric_cols,
        "pearson_r",
    ]
    table = model_df[[col for col in display_cols if col in model_df.columns]].copy()
    if "mean_inference_ms" in table.columns:
        table["mean_inference_ms"] = table["mean_inference_ms"] * 1000
    if winner_metric in table.columns:
        table = table.sort_values([winner_metric, "model"], na_position="last")
    return table.rename(
        columns={
            "model": "Model",
            "model_type": "Model type",
            "mean_inference_ms": "Mean inference (μs)",
            best_col: "# model wins",
            winner_metric: winner_metric.upper(),
            "qlike": "QLIKE",
            "mae": "MAE",
            "mse": "MSE",
            "mape": "MAPE",
            "rmspe": "RMSPE",
            "pearson_r": "Pearson r",
        }
    )


def render() -> None:
    st.title("Stock Universe")

    summary_df, corr_df, pca_df, pca_explained, model_comparison_df = _load_universe_data()

    st.caption("Cross-stock overview for volatility behaviour and model performance.")
    if summary_df.empty or corr_df.empty:
        st.info("Run models from the Model Specification tab to populate the universe view from backend artifacts.")
        return

    num_stocks = len(summary_df)
    avg_vol = summary_df["mean_volatility"].mean()
    most_volatile_stock = _stock_label(summary_df.loc[summary_df["mean_volatility"].idxmax(), "stock_id"])
    hardest_stock = _stock_label(summary_df.loc[summary_df["rmse"].idxmax(), "stock_id"])

    st.subheader("Universe Controls")
    c1, c2 = st.columns([1.1, 2.2])

    with c1:
        ranking_metric = st.selectbox(
            "Ranking metric",
            list(RANKING_METRICS),
            key="universe_ranking_metric",
        )
        manual_stock_input = st.text_area(
            "Manual stock list",
            placeholder="0, stock_1, stock_27",
            help="Adds these stocks to the automatic Top-N set. Duplicates are ignored.",
            key="universe_manual_stock_list",
            height=118,
        )

    with c2:
        top_controls, sort_controls = st.columns([1.4, 1])
        with top_controls:
            max_top_n = len(summary_df)
            top_n = st.slider(
                "Top-N stocks to display",
                min_value=1,
                max_value=max_top_n,
                value=max_top_n,
                key="universe_top_n",
            )
        with sort_controls:
            sort_order = st.radio(
                "Sort order",
                ["Descending", "Ascending"],
                horizontal=True,
                key="universe_sort_order",
            )

        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stocks", f"{num_stocks}")
        m2.metric("Average volatility", f"{avg_vol:.2f}")
        m3.metric("Most volatile", most_volatile_stock)
        m4.metric("Hardest to predict", hardest_stock)

    metric_col = RANKING_METRICS[ranking_metric]
    ascending = sort_order == "Ascending"

    ranked_df = summary_df.sort_values(by=metric_col, ascending=ascending).reset_index(drop=True)
    manual_stocks, missing_stocks = _parse_manual_stocks(manual_stock_input, summary_df["stock_id"].tolist())
    if missing_stocks:
        st.warning(f"These stocks are not available in the current run: {', '.join(missing_stocks)}")
    if manual_stock_input.strip() and not manual_stocks:
        st.info("Enter at least one stock from the current run, or clear the manual list to use only automatic Top-N ranking.")
        return

    top_df = _select_display_stocks(ranked_df, manual_stocks, top_n)
    selected_count = len(top_df)
    manual_additions = selected_count - top_n
    selected_label = f"Top {top_n} by {ranking_metric}"
    if manual_additions:
        selected_label = f"{selected_label} + {manual_additions} Manual"
    stock_ordered_df = _sort_by_stock_number(summary_df)

    st.subheader(selected_label)
    st.plotly_chart(
        _ranking_chart(top_df, metric_col, ranking_metric),
        width="stretch",
    )

    st.divider()

    st.subheader("Model Comparison")
    if model_comparison_df.empty:
        st.info("No model comparison data is available for this run.")
    else:
        st.dataframe(
            _model_comparison_view(model_comparison_df, "rmse"),
            hide_index=True,
            width="stretch",
            column_config={
                "Mean inference (μs)": st.column_config.NumberColumn(format="%.3f"),
                "# model wins": st.column_config.NumberColumn(format="%d"),
                "MSE": st.column_config.NumberColumn(format="%.3f"),
                "RMSE": st.column_config.NumberColumn(format="%.3f"),
                "MAE": st.column_config.NumberColumn(format="%.3f"),
                "MAPE": st.column_config.NumberColumn(format="%.3f"),
                "RMSPE": st.column_config.NumberColumn(format="%.3f"),
                "QLIKE": st.column_config.NumberColumn(format="%.3f"),
                "Pearson r": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    st.subheader("Stock Similarity")
    st.caption("Correlation view for the currently ranked stocks.")
    st.plotly_chart(
        _correlation_heatmap(corr_df, top_df["stock_id"].tolist()),
        width="stretch",
    )

    st.subheader("PCA Stock Map")
    pca_components = sorted(
        [col for col in pca_df.columns if col.startswith("PC")],
        key=lambda col: int(col.replace("PC", "")),
    )
    if len(pca_components) < 2:
        st.info("At least two PCA components are needed to plot the stock map.")
    else:
        pc_left, pc_right = st.columns(2)
        with pc_left:
            x_component = st.selectbox(
                "X principal component",
                pca_components,
                index=0,
                key="universe_pca_x_component",
            )
        with pc_right:
            y_component = st.selectbox(
                "Y principal component",
                pca_components,
                index=1,
                key="universe_pca_y_component",
            )
        if x_component == y_component:
            st.warning("Choose two different principal components.")
        else:
            st.plotly_chart(
                _pca_scatter(pca_df, summary_df, x_component, y_component, pca_explained),
                width="stretch",
            )

    left, right = st.columns([2.2, 1])

    with left:
        st.subheader("Per-Stock Summary")
        st.dataframe(
            _stock_summary_view(stock_ordered_df),
            hide_index=True,
            width='stretch',
            column_config={
                "Mean volatility": st.column_config.NumberColumn(format="%.6f"),
                "MSE": st.column_config.NumberColumn(format="%.6f"),
                "RMSE": st.column_config.NumberColumn(format="%.6f"),
                "MAE": st.column_config.NumberColumn(format="%.6f"),
                "MAPE": st.column_config.NumberColumn(format="%.3f"),
                "RMSPE": st.column_config.NumberColumn(format="%.6f"),
                "QLIKE": st.column_config.NumberColumn(format="%.5f"),
            },
        )
