import re

import pandas as pd
import streamlit as st

from back_end.service import available_stocks, get_latest_or_selected_run, load_individual_page_data
from charts import realised_vol_chart, realised_vs_predicted_scatter
from stock_registry import ALL_BOOK_STEMS


_STOCK_ID_RE = re.compile(r"^stock_(\d+)$")


def _stock_sort_key(stock_id: str) -> tuple[int, int, str]:
    match = _STOCK_ID_RE.match(stock_id)
    if match:
        return (0, int(match.group(1)), stock_id)
    return (1, 0, stock_id)


def _numeric_sort_key(value: object) -> tuple[int, float, str]:
    try:
        return (0, float(value), str(value))
    except (TypeError, ValueError):
        return (1, 0, str(value))


def _sort_stock_options(stock_ids: list[str]) -> list[str]:
    return sorted(stock_ids, key=_stock_sort_key)


def _sort_time_options(time_ids: list[int]) -> list[int]:
    return sorted(time_ids, key=_numeric_sort_key)


def render() -> None:
    st.title("Individual Stock")

    run_id = get_latest_or_selected_run(st.session_state.get("selected_run_id"))
    all_stocks = _sort_stock_options(available_stocks() or ALL_BOOK_STEMS)
    default_idx = 0
    saved = st.session_state.get("selected_stock")
    if saved and saved in all_stocks:
        default_idx = all_stocks.index(saved)

    col_stock, col_time = st.columns([3, 1])

    with col_stock:
        stock_id = st.selectbox("Select stock", all_stocks, index=default_idx, key="individual_stock_select")
        st.session_state.selected_stock = stock_id

    with col_time:
        first_payload = load_individual_page_data(run_id, stock_id)
        time_options = _sort_time_options(first_payload["time_ids"] or list(range(1, 11)))
        saved_time = st.session_state.get("individual_time_select")
        time_index = time_options.index(saved_time) if saved_time in time_options else 0
        time_id = st.selectbox(
            "Time window (10-min period)",
            options=time_options,
            index=time_index,
            key="individual_time_select",
        )

    payload = load_individual_page_data(run_id, stock_id, int(time_id))
    if run_id:
        st.caption(f"Showing backend run `{run_id}`.")
    else:
        st.info("Run models from the Model Specification tab to replace the demo chart with backend predictions.")

    st.plotly_chart(
        realised_vol_chart(
            stock_id,
            int(time_id),
            payload.get("realized_series"),
            payload.get("prediction_curves"),
        ),
        width="stretch",
    )

    st.plotly_chart(
        realised_vs_predicted_scatter(
            stock_id,
            payload.get("stock_predictions"),
        ),
        width="stretch",
    )

    st.subheader("Model Performance")
    stock_match = _STOCK_ID_RE.match(stock_id)
    stock_label = stock_match.group(1) if stock_match else stock_id
    st.caption(f"Prediction metrics over all Time IDs for Stock {stock_label}")

    df = pd.DataFrame(payload["model_metrics"]).rename(columns={
        "model":        "Model",
        "inference_us": "Inference time (μs)",
        "rmse":         "RMSE",
        "qlike":        "QLIKE",
    })
    for col in ["Inference time (μs)", "RMSE", "QLIKE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Inference time (μs)": st.column_config.NumberColumn(format="%.3f μs"),
            "RMSE":           st.column_config.NumberColumn(format="%.6f"),
            "QLIKE":          st.column_config.NumberColumn(format="%.5f"),
        },
    )
