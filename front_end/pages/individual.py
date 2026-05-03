import pandas as pd
import streamlit as st

from charts import realised_vol_chart
from stock_registry import ALL_BOOK_STEMS

MODEL_METRICS = [
    {"model": "LSTM",        "inference_us": 2340, "rmse": 0.000312, "qlike": 0.01847, "pred_target": 0.001452},
    {"model": "GRU",         "inference_us": 1870, "rmse": 0.000298, "qlike": 0.01763, "pred_target": 0.001452},
    {"model": "XGBoost",     "inference_us":  410, "rmse": 0.000341, "qlike": 0.02105, "pred_target": 0.001452},
    {"model": "HAR-RV",      "inference_us":   95, "rmse": 0.000367, "qlike": 0.02289, "pred_target": 0.001452},
    {"model": "GARCH(1,1)",  "inference_us":   58, "rmse": 0.000403, "qlike": 0.02514, "pred_target": 0.001452},
    {"model": "Transformer", "inference_us": 3180, "rmse": 0.000287, "qlike": 0.01698, "pred_target": 0.001452},
]


def render() -> None:
    st.title("Individual Stock")

    all_stocks = ALL_BOOK_STEMS
    default_idx = 0
    saved = st.session_state.get("selected_stock")
    if saved and saved in all_stocks:
        default_idx = all_stocks.index(saved)

    col_stock, col_time = st.columns([3, 1])

    with col_stock:
        stock_id = st.selectbox("Select stock", all_stocks, index=default_idx, key="individual_stock_select")
        st.session_state.selected_stock = stock_id

    with col_time:
        time_id = st.selectbox(
            "Time window (10-min period)",
            options=list(range(1, 11)),
            index=0,
            key="individual_time_select",
        )

    st.plotly_chart(
        realised_vol_chart(stock_id, time_id),
        use_container_width=True,
    )

    st.subheader("Model Performance")
    st.caption("Prediction metrics over the last 5-minute window · inference time per prediction call")

    df = pd.DataFrame(MODEL_METRICS).rename(columns={
        "model":        "Model",
        "inference_us": "Inference time (μs)",
        "rmse":         "RMSE",
        "qlike":        "QLIKE",
        "pred_target":  "Mean 5-min Vol",
    })
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Inference time (μs)": st.column_config.NumberColumn(format="%d μs"),
            "RMSE":           st.column_config.NumberColumn(format="%.6f"),
            "QLIKE":          st.column_config.NumberColumn(format="%.5f"),
            "Mean 5-min Vol": st.column_config.NumberColumn(format="%.6f"),
        },
    )
