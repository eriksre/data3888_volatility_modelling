import streamlit as st

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="Optiver Volatility Dashboard",
    layout="wide"
)

# --------------------------
# Sidebar
# --------------------------
st.sidebar.title("Controls")

page = st.sidebar.selectbox(
    "Select Page",
    ["Overview", "Model Performance"]
)

stock_id = st.sidebar.selectbox(
    "Select Stock ID",
    ["stock_id list"]
)

time_range = st.sidebar.slider(
    "Select Time Range",
    min_value=0,
    max_value=100,
    value=(0, 100)
)

feature = st.sidebar.selectbox(
    "Select Feature",
    ["wap", "spread", "volume", "log_return", "imbalance","xxxx","xxxxx"]
)

# --------------------------
# Main content blablabla
# --------------------------
st.title("Optiver Volatility Dashboard")

if page == "Overview":
    st.header("Market Overview")
    st.write("This page shows market behaviour over time.")

    # KPI cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Mean WAP")
        st.info("Here should be mean WAP value")

    with col2:
        st.subheader("Mean Spread")
        st.info("Here should be mean spread value")

    with col3:
        st.subheader("Mean Volatility")
        st.info("Here should be mean realised volatility")

    st.divider()

    # Main chart placeholder
    st.subheader("WAP over Time")
    st.info("Here should be a line chart of WAP over time")

    st.divider()

    # Secondary charts
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Spread over Time")
        st.info("Here should be a line chart of spread over time")

    with col5:
        st.subheader("Volume over Time")
        st.info("Here should be a line chart of trading volume over time")

    st.divider()

    # Insight box
    st.subheader("Insight")
    st.info("Here should be explanation of how spread / volume relates to volatility")


elif page == "Model Performance":
    st.header("Model Performance")
    st.write("This page evaluates model predictions.")

    # KPI cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("RMSE")
        st.info("Here should be RMSE value")

    with col2:
        st.subheader("MAE")
        st.info("Here should be MAE value")

    with col3:
        st.subheader("Correlation / R²")
        st.info("Here should be correlation or R²")

    st.divider()

    # Main chart placeholder
    st.subheader("Actual vs Predicted Volatility")
    st.info("Here should be a line chart comparing actual and predicted volatility")

    st.divider()

    # Secondary charts
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Residual Plot")
        st.info("Here should be residual over time (predicted - actual)")

    with col5:
        st.subheader("Residual Distribution")
        st.info("Here should be histogram of residuals")

    st.divider()

    # Insight box
    st.subheader("Model Insight")
    st.info("Here should be explanation of model performance and limitations")