import pandas as pd
import streamlit as st

from back_end.service import available_stocks, get_latest_or_selected_run, load_individual_page_data
from charts import realised_vol_chart
from stock_registry import ALL_BOOK_STEMS


def render() -> None:
    st.title("Individual Stock")

    run_id = get_latest_or_selected_run(st.session_state.get("selected_run_id"))
    all_stocks = available_stocks() or ALL_BOOK_STEMS
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
        time_options = first_payload["time_ids"] or list(range(1, 11))
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

    st.subheader("Model Performance")
    st.caption("Prediction metrics over the last 5-minute window · inference time per prediction call")

    df = pd.DataFrame(payload["model_metrics"]).rename(columns={
        "model":        "Model",
        "inference_us": "Inference time (μs)",
        "rmse":         "RMSE",
        "qlike":        "QLIKE",
        "pred_target":  "Mean 5-min Vol",
    })
    for col in ["Inference time (μs)", "RMSE", "QLIKE", "Mean 5-min Vol"]:
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
            "Mean 5-min Vol": st.column_config.NumberColumn(format="%.6f"),
        },
    )
