import numpy as np
import plotly.graph_objects as go


def realised_vol_chart(stock_id: str, time_id: int = 1) -> go.Figure:
    """Dummy realised volatility chart: 5 min actual + 5 min predicted."""
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
        title=dict(text=f"Realised Volatility — {stock_id}  ·  Window {time_id}", font=dict(size=16)),
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=80, b=50),
        hovermode="x unified",
    )
    return fig
