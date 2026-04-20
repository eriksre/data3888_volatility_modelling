import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Universe Page Demo",
    layout="wide"
)

# ---------------------------------------------------
# use Dummy data
# ---------------------------------------------------
@st.cache_data
def load_dummy_universe_data():
    np.random.seed(42)

    stocks = [f"stock_{i}" for i in range(20)]

    summary_df = pd.DataFrame({
        "stock_id": stocks,
        "mean_volatility": np.round(np.random.uniform(0.0015, 0.0050, len(stocks)), 6),
        "rmse": np.round(np.random.uniform(0.00015, 0.00060, len(stocks)), 6),
        "qlike": np.round(np.random.uniform(0.010, 0.030, len(stocks)), 5),
        "best_model": np.random.choice(
            ["Random Forest", "LASSO", "GARCH", "XGBoost"], len(stocks)
        )
    })

    n = len(stocks)
    A = np.random.uniform(-0.2, 0.95, size=(n, n))
    corr = (A + A.T) / 2
    np.fill_diagonal(corr, 1.0)
    corr_df = pd.DataFrame(corr, index=stocks, columns=stocks)

    return summary_df, corr_df


summary_df, corr_df = load_dummy_universe_data()

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("Stock Universe")
st.caption("Cross-stock overview for volatility behaviour and model performance.")

# ---------------------------------------------------
# Controls
# ---------------------------------------------------
with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Universe Controls")

    c1, c2, c3 = st.columns([1.2, 1.2, 1])

    with c1:
        ranking_metric = st.selectbox(
            "Ranking metric",
            ["Mean Volatility", "RMSE", "QLIKE"]
        )

    with c2:
        top_n = st.slider(
            "Top-N stocks to display",
            min_value=5,
            max_value=len(summary_df),
            value=10
        )

    with c3:
        sort_order = st.radio(
            "Sort order",
            ["Descending", "Ascending"],
            horizontal=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

metric_map = {
    "Mean Volatility": "mean_volatility",
    "RMSE": "rmse",
    "QLIKE": "qlike"
}
metric_col = metric_map[ranking_metric]
ascending = sort_order == "Ascending"

ranked_df = summary_df.sort_values(by=metric_col, ascending=ascending).reset_index(drop=True)
top_df = ranked_df.head(top_n)

# ---------------------------------------------------
# Summary cards
# ---------------------------------------------------
num_stocks = len(summary_df)
avg_vol = summary_df["mean_volatility"].mean()
most_volatile_stock = summary_df.loc[summary_df["mean_volatility"].idxmax(), "stock_id"]
hardest_stock = summary_df.loc[summary_df["rmse"].idxmax(), "stock_id"]

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="custom-card">
        <div class="card-label">Number of Stocks</div>
        <div class="card-value">{num_stocks}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="custom-card">
        <div class="card-label">Average Volatility</div>
        <div class="card-value">{avg_vol:.6f}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="custom-card">
        <div class="card-label">Most Volatile Stock</div>
        <div class="card-value">{most_volatile_stock}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="custom-card">
        <div class="card-label">Hardest to Predict</div>
        <div class="card-value">{hardest_stock}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------
# Ranking chart
# ---------------------------------------------------
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader(f"Top {top_n} Stocks by {ranking_metric}")

fig_bar = px.bar(
    top_df,
    x="stock_id",
    y=metric_col,
    color="best_model",
    labels={
        "stock_id": "Stock",
        metric_col: ranking_metric,
        "best_model": "Best Model"
    },
    title=f"Stock Ranking by {ranking_metric}",
    color_discrete_sequence=["#ff6b6b", "#f59e0b", "#60a5fa", "#34d399"]
)

fig_bar.update_layout(
    height=500,
    xaxis_tickangle=-45,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(color="#f3f4f6"),
    legend_title_text="Best Model",
    margin=dict(l=20, r=20, t=60, b=20)
)

fig_bar.update_xaxes(showgrid=False, color="#d1d5db")
fig_bar.update_yaxes(gridcolor="rgba(255,255,255,0.08)", color="#d1d5db")

st.plotly_chart(fig_bar, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Heatmap
# ---------------------------------------------------
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Stock Similarity View")
st.write(
    "This heatmap shows "
)

fig_heat = px.imshow(
    corr_df,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
    title="Cross-Stock Correlation Heatmap"
)

fig_heat.update_layout(
    height=700,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(color="#f3f4f6"),
    margin=dict(l=20, r=20, t=60, b=20),
    coloraxis_colorbar=dict(title="Corr")
)

fig_heat.update_xaxes(color="#d1d5db")
fig_heat.update_yaxes(color="#d1d5db")

st.plotly_chart(fig_heat, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Table + stock selection
# ---------------------------------------------------
left, right = st.columns([2.2, 1])

with left:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Per-Stock Summary")
    st.dataframe(
        ranked_df,
        use_container_width=True,
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Open Individual Stock View")
    selected_stock = st.selectbox(
        "Select a stock",
        options=summary_df["stock_id"].tolist()
    )

    if st.button("Open selected stock"):
        st.success(f"Here this would navigate to the Individual Stock page for {selected_stock}.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Insight
# ---------------------------------------------------
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Universe Insight")
st.markdown("""
<div class="insight-box">
Functions: 1. Helps clients compare stocks globally, highlighting which stocks have the greatest volatility in the ranking table; similarity can find stocks with similar trends, and suitable models can be used (similar stocks can use similar models and can be grouped together for display; currently, it's relatively rudimentary, and I haven't found a better usable chart yet).
--Example:
High RMSE + High QLIKE: Stocks are difficult to predict.
Low RMSE but High QLIKE: Stocks have small average error, poor risk structure.
balbalabla

2. Can redirect to individual stock pages.
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)