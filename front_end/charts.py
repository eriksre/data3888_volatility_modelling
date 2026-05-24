import numpy as np
import pandas as pd
import plotly.graph_objects as go


MODEL_COLORS = [
    "#ffb703",
    "#8ecae6",
    "#fb8500",
    "#90be6d",
    "#f15bb5",
    "#00bbf9",
    "#fee440",
    "#9b5de5",
]
SCATTER_DISPLAY_QUANTILE = 0.995
SCATTER_DISPLAY_IQR_MULTIPLIER = 50
REALIZED_VOL_TOP_MARGIN = 125
REALIZED_VOL_LEGEND_Y = 1.08


def _empty_chart(stock_id: str, time_id: int) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=f"Realised Volatility — {stock_id}  ·  Window {time_id}",
            font=dict(size=16),
            y=0.98,
            yanchor="top",
        ),
        xaxis_title="Time (seconds)",
        yaxis_title="Realised Volatility",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=REALIZED_VOL_TOP_MARGIN, b=50),
    )
    return fig


def _demo_chart(stock_id: str, time_id: int = 1) -> go.Figure:
    """Fallback chart shown before a backend run exists."""
    seed = abs(hash((stock_id, time_id))) % (2**32)
    rng = np.random.default_rng(seed)

    # 5-second buckets across 10 minutes → 120 points (0, 5, 10, …, 595 s)
    times = np.arange(0, 600, 5)
    split = 300  # seconds — boundary between actual and predicted

    # Simulate a mean-reverting realised vol series
    base_vol = 0.0015 + rng.uniform(0, 0.001)
    noise = rng.normal(0, 0.00015, size=len(times))
    rv = np.zeros(len(times))
    rv[0] = base_vol
    for i in range(1, len(times)):
        rv[i] = 0.85 * rv[i - 1] + 0.15 * base_vol + noise[i]
    rv = np.clip(rv, 0.0002, None)

    actual_mask = times < split
    pred_mask = ~actual_mask

    t_actual = times[actual_mask]
    rv_actual = rv[actual_mask]

    t_pred = times[pred_mask]
    rv_pred = rv[pred_mask]
    # slight upward drift + widening uncertainty band for predicted
    drift = np.linspace(0, 0.0004, len(t_pred))
    rv_pred_shifted = rv_pred + drift
    sigma = np.linspace(0.0002, 0.0008, len(t_pred))
    rv_upper = rv_pred_shifted + sigma
    rv_lower = np.clip(rv_pred_shifted - sigma, 0.0002, None)

    fig = go.Figure()

    # Shaded backgrounds
    fig.add_vrect(
        x0=0, x1=split,
        fillcolor="steelblue", opacity=0.06,
        layer="below", line_width=0,
    )
    fig.add_vrect(
        x0=split, x1=595,
        fillcolor="darkorange", opacity=0.06,
        layer="below", line_width=0,
    )

    # Confidence band for predicted
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_pred, t_pred[::-1]]),
        y=np.concatenate([rv_upper, rv_lower[::-1]]),
        fill="toself",
        fillcolor="rgba(255,140,0,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="Prediction band",
    ))

    # Actual RV line
    fig.add_trace(go.Scatter(
        x=t_actual,
        y=rv_actual,
        mode="lines",
        line=dict(color="steelblue", width=2),
        name="Realised Volatility (actual)",
    ))

    # Predicted RV line — bridge from last actual point
    bridge_x = np.concatenate([[t_actual[-1]], t_pred])
    bridge_y = np.concatenate([[rv_actual[-1]], rv_pred_shifted])
    fig.add_trace(go.Scatter(
        x=bridge_x,
        y=bridge_y,
        mode="lines",
        line=dict(color="darkorange", width=2, dash="dash"),
        name="Realised Volatility (predicted)",
    ))

    # Vertical split line
    fig.add_vline(
        x=split,
        line_width=1.5, line_dash="dot", line_color="grey",
        annotation_text="Forecast →",
        annotation_position="top right",
        annotation_font=dict(size=11, color="grey"),
    )

    fig.update_layout(
        title=dict(
            text=f"Realised Volatility — {stock_id}  ·  Window {time_id}",
            font=dict(size=16),
            y=0.98,
            yanchor="top",
        ),
        xaxis=dict(
            title="Time (seconds)",
            tickvals=list(range(0, 601, 60)),
            ticktext=[f"{t//60}m" for t in range(0, 601, 60)],
            showgrid=True, gridcolor="rgba(200,200,200,0.3)",
        ),
        yaxis=dict(
            title="Realised Volatility",
            tickformat=".4f",
            showgrid=True, gridcolor="rgba(200,200,200,0.3)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=REALIZED_VOL_LEGEND_Y, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=REALIZED_VOL_TOP_MARGIN, b=50),
        hovermode="x unified",
    )
    return fig


def realised_vol_chart(
    stock_id: str,
    time_id: int = 1,
    realized_series: pd.DataFrame | None = None,
    prediction_curves: pd.DataFrame | None = None,
) -> go.Figure:
    if realized_series is None or realized_series.empty:
        return _demo_chart(stock_id, time_id)

    fig = _empty_chart(stock_id, time_id)
    realized = realized_series.sort_values("seconds_in_bucket")
    split_second = int(realized["split_second"].dropna().iloc[0]) if "split_second" in realized and realized["split_second"].notna().any() else 300
    max_second = int(realized["seconds_in_bucket"].max())

    fig.add_vrect(
        x0=0,
        x1=split_second,
        fillcolor="steelblue",
        opacity=0.06,
        layer="below",
        line_width=0,
    )
    fig.add_vrect(
        x0=split_second,
        x1=max_second,
        fillcolor="darkorange",
        opacity=0.06,
        layer="below",
        line_width=0,
    )

    observed = realized[realized["segment"] == "observed"]
    heldout = realized[realized["segment"] == "heldout_actual"]
    if not observed.empty:
        fig.add_trace(
            go.Scatter(
                x=observed["seconds_in_bucket"],
                y=observed["realized_vol"],
                mode="lines",
                line=dict(color="#4dabf7", width=2.5),
                name="Observed realised volatility",
            )
        )
    if not heldout.empty:
        bridge = pd.concat([observed.tail(1), heldout], ignore_index=True) if not observed.empty else heldout
        fig.add_trace(
            go.Scatter(
                x=bridge["seconds_in_bucket"],
                y=bridge["realized_vol"],
                mode="lines",
                line=dict(color="#dee2e6", width=2.5, dash="dot"),
                name="Held-out realised volatility",
            )
        )

    if prediction_curves is not None and not prediction_curves.empty:
        for idx, (model, frame) in enumerate(prediction_curves.groupby("model", sort=False)):
            frame = frame.sort_values("seconds_in_bucket")
            kind = frame["prediction_kind"].iloc[0] if "prediction_kind" in frame else "horizon_line"
            fig.add_trace(
                go.Scatter(
                    x=frame["seconds_in_bucket"],
                    y=frame["pred_vol"],
                    mode="lines",
                    line=dict(
                        color=MODEL_COLORS[idx % len(MODEL_COLORS)],
                        width=2.2,
                        dash="solid" if kind == "garch_path" else "dash",
                    ),
                    name=f"{model} forecast",
                )
            )

    fig.add_vline(
        x=split_second,
        line_width=1.5,
        line_dash="dot",
        line_color="#adb5bd",
        annotation_text="Forecast start",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#adb5bd"),
    )
    fig.update_layout(
        xaxis=dict(
            title="Time (seconds)",
            tickvals=list(range(0, 601, 60)),
            ticktext=[f"{t//60}m" for t in range(0, 601, 60)],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.22)",
        ),
        yaxis=dict(
            title="Realised Volatility",
            tickformat=".4f",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.22)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=REALIZED_VOL_LEGEND_Y, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def realised_vs_predicted_scatter(stock_id: str, predictions: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=520,
        title=dict(text=f"Realised vs Predicted Volatility — {stock_id}", font=dict(size=16)),
        xaxis=dict(
            title="Realised Volatility",
            tickformat=".4f",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.22)",
        ),
        yaxis=dict(
            title="Predicted Volatility",
            tickformat=".4f",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.22)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=80, b=50),
    )

    if predictions is None or predictions.empty:
        fig.add_annotation(
            text="No backend predictions available for this stock.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#adb5bd"),
        )
        return fig

    required = {"model", "actual_vol", "pred_vol"}
    if not required.issubset(predictions.columns):
        fig.add_annotation(
            text="Prediction data is missing realised or predicted volatility columns.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#adb5bd"),
        )
        return fig

    scatter_data = predictions.copy()
    scatter_data["actual_vol"] = pd.to_numeric(scatter_data["actual_vol"], errors="coerce")
    scatter_data["pred_vol"] = pd.to_numeric(scatter_data["pred_vol"], errors="coerce")
    scatter_data = scatter_data.dropna(subset=["actual_vol", "pred_vol"])
    scatter_data = scatter_data[np.isfinite(scatter_data["actual_vol"]) & np.isfinite(scatter_data["pred_vol"])]

    if scatter_data.empty:
        fig.add_annotation(
            text="No finite realised/predicted volatility pairs are available for this stock.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#adb5bd"),
        )
        return fig

    scatter_values = scatter_data[["actual_vol", "pred_vol"]].to_numpy(dtype=float).ravel()
    quantile_cap = float(np.quantile(scatter_values, SCATTER_DISPLAY_QUANTILE))
    q1, q3 = np.quantile(scatter_values, [0.25, 0.75])
    iqr = float(q3 - q1)
    center = float(np.median(scatter_values))
    robust_step = max(iqr, abs(center) * 0.01, 0.001)
    robust_cap = center + SCATTER_DISPLAY_IQR_MULTIPLIER * robust_step
    display_cap = min(quantile_cap, robust_cap)
    display_cap = max(display_cap, float(scatter_data["actual_vol"].max()))
    display_mask = (scatter_data["actual_vol"] <= display_cap) & (scatter_data["pred_vol"] <= display_cap)
    display_data = scatter_data[display_mask]
    hidden_count = len(scatter_data) - len(display_data)
    if display_data.empty:
        display_data = scatter_data
        hidden_count = 0

    max_vol = float(display_data[["actual_vol", "pred_vol"]].max().max())
    padding = max(max_vol * 0.05, 0.001)
    axis_min = 0.0
    axis_max = max_vol + padding

    fig.add_trace(
        go.Scattergl(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            line=dict(color="#adb5bd", width=1.5, dash="dot"),
            name="Perfect prediction",
            hoverinfo="skip",
        )
    )

    has_hover_context = {"time_id", "fold"}.issubset(display_data.columns)
    for idx, (model, frame) in enumerate(display_data.groupby("model", sort=False)):
        fig.add_trace(
            go.Scattergl(
                x=frame["actual_vol"],
                y=frame["pred_vol"],
                mode="markers",
                marker=dict(
                    color=MODEL_COLORS[idx % len(MODEL_COLORS)],
                    size=8,
                    opacity=0.78,
                    line=dict(width=0.5, color="rgba(255,255,255,0.55)"),
                ),
                name=str(model),
                customdata=frame[["time_id", "fold"]].to_numpy() if has_hover_context else None,
                hovertemplate=(
                    "Realised: %{x:.6f}<br>"
                    "Predicted: %{y:.6f}<br>"
                    "Time ID: %{customdata[0]}<br>"
                    "Fold: %{customdata[1]}<extra>%{fullData.name}</extra>"
                    if has_hover_context
                    else "Realised: %{x:.6f}<br>Predicted: %{y:.6f}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.update_xaxes(range=[axis_min, axis_max])
    fig.update_yaxes(range=[axis_min, axis_max])
    return fig


def pca_variance_explained_chart(variance: pd.DataFrame | None, n_components: int) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=280,
        title=dict(text="Variance Explained by Principal Component", font=dict(size=15)),
        xaxis=dict(title="Principal component", showgrid=False),
        yaxis=dict(
            title="Proportion of variance explained",
            tickformat=".0%",
            range=[0, 1],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.22)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=55, b=45, l=20, r=20),
        showlegend=False,
    )

    if variance is None or variance.empty:
        fig.add_annotation(
            text="PCA variance data is not available.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#adb5bd"),
        )
        return fig

    plot_df = variance.head(int(n_components)).copy()
    max_ratio = float(plot_df["explained_variance_ratio"].max())
    fig.update_yaxes(range=[0, min(1.0, max(max_ratio * 1.15, 0.05))])
    fig.add_trace(
        go.Bar(
            x=plot_df["component"],
            y=plot_df["explained_variance_ratio"],
            marker_color="#4dabf7",
            hovertemplate="%{x}<br>Variance explained: %{y:.2%}<extra></extra>",
        )
    )
    return fig


def pca_cumulative_variance_chart(variance: pd.DataFrame | None, n_components: int) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=280,
        title=dict(text="Cumulative Variance Explained", font=dict(size=15)),
        xaxis=dict(title="Principal component", showgrid=False),
        yaxis=dict(
            title="Cumulative variance",
            tickformat=".0%",
            range=[0, 1],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.22)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=55, b=45, l=20, r=20),
        showlegend=False,
    )

    if variance is None or variance.empty:
        fig.add_annotation(
            text="PCA variance data is not available.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#adb5bd"),
        )
        return fig

    plot_df = variance.head(int(n_components)).copy()
    plot_df["cumulative_variance_ratio"] = plot_df["explained_variance_ratio"].cumsum()
    fig.add_trace(
        go.Scatter(
            x=plot_df["component"],
            y=plot_df["cumulative_variance_ratio"],
            mode="lines+markers",
            line=dict(color="#f59f00", width=2.5),
            marker=dict(size=7, color="#f59f00"),
            hovertemplate="%{x}<br>Cumulative variance explained: %{y:.2%}<extra></extra>",
        )
    )
    return fig
