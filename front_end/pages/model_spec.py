import streamlit as st

# ---------------------------------------------------------------------------
# Static reference data (dummy for now)
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = [
    "Transformer",
    "LSTM",
    "GRU",
    "XGBoost",
    "Random Forest",
    "HAR-RV",
    "GARCH(1,1)",
    "Linear Regression",
]

ALL_FEATURES = [
    "WAP Returns",
    "Bid-Ask Spread",
    "Order Book Imbalance",
    "Rolling RV (5 s)",
    "Rolling RV (30 s)",
    "Rolling RV (60 s)",
    "Rolling RV (300 s)",
    "Depth Ratio (Level 1)",
    "Depth Ratio (Levels 1-3)",
    "Volume Imbalance",
    "Mid-Price Change",
    "Time of Day",
    "Day of Week",
    "Lagged RV (1-step)",
    "Lagged RV (5-step)",
]

AUTO_FEATURES: dict[str, list[str]] = {
    "Transformer":        ["WAP Returns", "Rolling RV (5 s)", "Rolling RV (30 s)", "Order Book Imbalance", "Bid-Ask Spread"],
    "LSTM":               ["WAP Returns", "Rolling RV (5 s)", "Rolling RV (30 s)", "Order Book Imbalance", "Bid-Ask Spread"],
    "GRU":                ["WAP Returns", "Rolling RV (5 s)", "Rolling RV (30 s)", "Order Book Imbalance", "Bid-Ask Spread"],
    "XGBoost":            ["Lagged RV (1-step)", "Lagged RV (5-step)", "Rolling RV (60 s)", "Bid-Ask Spread", "Depth Ratio (Level 1)", "Time of Day"],
    "Random Forest":      ["Lagged RV (1-step)", "Lagged RV (5-step)", "Rolling RV (60 s)", "Bid-Ask Spread", "Depth Ratio (Level 1)", "Time of Day"],
    "HAR-RV":             ["Rolling RV (5 s)", "Rolling RV (300 s)", "Lagged RV (1-step)", "Lagged RV (5-step)"],
    "GARCH(1,1)":         ["Mid-Price Change"],
    "Linear Regression":  ["Lagged RV (1-step)", "Lagged RV (5-step)", "Rolling RV (60 s)"],
}

LOSS_FUNCTIONS = ["RMSE", "RMSPE", "QLIKE", "MAE", "MAPE", "Huber"]

LOSS_DESCRIPTIONS = {
    "RMSE":  r"\sqrt{\frac{1}{N}\sum(\hat\sigma_t - \sigma_t)^2}",
    "RMSPE": r"\sqrt{\frac{1}{N}\sum\!\left(\frac{\hat\sigma_t - \sigma_t}{\sigma_t}\right)^{\!2}}",
    "QLIKE": r"\frac{\hat\sigma_t^2}{\sigma_t^2} - \log\frac{\hat\sigma_t^2}{\sigma_t^2} - 1",
    "MAE":   r"\frac{1}{N}\sum|\hat\sigma_t - \sigma_t|",
    "MAPE":  r"\frac{100}{N}\sum\left|\frac{\hat\sigma_t - \sigma_t}{\sigma_t}\right|",
    "Huber": r"\begin{cases}\tfrac12(\hat\sigma_t-\sigma_t)^2 & |\text{err}|\le\delta\\\delta|\text{err}|-\tfrac12\delta^2 & \text{otherwise}\end{cases}",
}

# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "model_list" not in st.session_state:
        st.session_state.model_list = []
    if "builder_feature_mode" not in st.session_state:
        st.session_state.builder_feature_mode = "Auto"


def _model_type_icon(model_type: str) -> str:
    icons = {
        "Transformer": "🤖",
        "LSTM": "🔁",
        "GRU": "🔂",
        "XGBoost": "🌲",
        "Random Forest": "🌳",
        "HAR-RV": "📐",
        "GARCH(1,1)": "📊",
        "Linear Regression": "📏",
    }
    return icons.get(model_type, "⚙️")


# ---------------------------------------------------------------------------
# Sub-sections
# ---------------------------------------------------------------------------

def _render_builder() -> None:
    """Configures and adds a model to the list.

    Uses plain widgets (no st.form) so the Auto/Manual radio triggers an
    immediate rerender without waiting for a submit button.  A 'form version'
    counter in session state is incremented on submit to reset all widget keys.
    """
    st.subheader("Configure a Model")

    # Version counter — incrementing it changes every widget key, which resets
    # all values back to their defaults (equivalent to clear_on_submit).
    v = st.session_state.get("builder_v", 0)

    # ---- Model type + name ----
    c1, c2 = st.columns([1, 1])
    with c1:
        model_type = st.selectbox("Model architecture", AVAILABLE_MODELS, key=f"mt_{v}")
    with c2:
        model_name = st.text_input(
            "Model name (optional)",
            placeholder="Leave blank to auto-name",
            key=f"mn_{v}",
        )

    st.divider()

    # ---- Prediction target ----
    st.markdown("**Prediction target**")
    pred_seconds = st.slider(
        "Forecast horizon (seconds of volatility to predict)",
        min_value=30, max_value=300, value=60, step=30,
        key=f"ps_{v}",
        help="The model will predict mean realised volatility over this trailing window.",
    )
    st.caption(f"Target: mean RV over the last **{pred_seconds} s** of the evaluation window")

    st.divider()

    # ---- Features ----
    st.markdown("**Features**")
    feature_mode = st.radio(
        "Feature selection",
        ["Auto", "Manual"],
        horizontal=True,
        key=f"fm_{v}",
        help="Auto: choose how many features to select automatically. Manual: pick specific features by name.",
    )

    if feature_mode == "Auto":
        n_features = st.slider(
            "Number of features to auto-select",
            min_value=1, max_value=30, value=5, step=1,
            key=f"nf_{v}",
            help="The top N features will be ranked and selected at training time.",
        )
        selected_features = list(range(n_features))
        st.caption(f"{n_features} feature{'s' if n_features != 1 else ''} will be selected automatically at training time.")
    else:
        selected_features = st.multiselect(
            "Select features",
            options=ALL_FEATURES,
            default=AUTO_FEATURES.get("LSTM", ALL_FEATURES[:5]),
            key=f"sf_{v}",
        )
        if not selected_features:
            st.warning("Select at least one feature.")

    st.divider()

    # ---- Train / test split ----
    st.markdown("**Train / test split**")
    split_options = [f"{t} / {100-t} %" for t in range(50, 100, 10)]
    split_label = st.select_slider(
        "Training proportion",
        options=split_options,
        value="70 / 30 %",
        key=f"sl_{v}",
    )
    train_pct = int(split_label.split(" / ")[0])
    n_folds = round(1 / (1 - train_pct / 100))
    st.markdown(
        f'<div style="text-align:right; margin-top:-0.5rem;">'
        f'<span style="'
        f'display:inline-block; padding:0.3rem 0.9rem; border-radius:6px;'
        f'border:1px solid rgba(128,128,128,0.3);'
        f'font-size:0.85rem; font-weight:500;">'
        f'{n_folds}-fold cross-validation'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- Loss functions ----
    st.markdown("**Loss function(s)**")
    selected_losses = st.multiselect(
        "Evaluation metrics",
        options=LOSS_FUNCTIONS,
        default=["RMSE", "QLIKE"],
        key=f"lo_{v}",
    )

    use_custom = st.checkbox("Add a custom loss function", key=f"uc_{v}")
    custom_loss_name = ""
    custom_loss_expr = ""
    if use_custom:
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            custom_loss_name = st.text_input(
                "Custom loss name", placeholder="e.g. Weighted RMSE", key=f"cln_{v}"
            )
        with cc2:
            custom_loss_expr = st.text_input(
                "Expression / parameters",
                placeholder="e.g. w1=2.0, w2=0.5  or  lambda y, yhat: ...",
                key=f"cle_{v}",
            )

    st.divider()

    if st.button("➕  Add model to list", type="primary", use_container_width=True):
        if feature_mode == "Manual" and not selected_features:
            st.error("Cannot add model: no features selected.")
            return

        name = model_name.strip() or f"{model_type} #{len(st.session_state.model_list) + 1}"
        losses = list(selected_losses)
        if use_custom and custom_loss_name:
            losses.append(f"{custom_loss_name} (custom)")

        entry = {
            "name": name,
            "type": model_type,
            "feature_mode": feature_mode,
            "features": selected_features,
            "pred_seconds": pred_seconds,
            "train_pct": train_pct,
            "n_folds": n_folds,
            "losses": losses,
            "custom_loss": {"name": custom_loss_name, "expr": custom_loss_expr} if (use_custom and custom_loss_name) else None,
        }
        st.session_state.model_list.append(entry)
        st.session_state.builder_v = v + 1  # reset all widget values
        st.rerun()


def _render_model_list() -> None:
    """Display the list of configured models with summary cards."""
    models = st.session_state.model_list

    if not models:
        st.info("No models configured yet. Use the form above to add one.")
        return

    st.subheader(f"Model List  ({len(models)} configured)")

    for i, m in enumerate(models):
        icon = _model_type_icon(m["type"])
        with st.expander(f"{icon}  **{m['name']}**  —  {m['type']}", expanded=True):
            col_left, col_right, col_del = st.columns([3, 3, 1])

            with col_left:
                st.markdown(f"**Forecast horizon:** {m['pred_seconds']} s")
                st.markdown(f"**Train / test split:** {m['train_pct']} / {100 - m['train_pct']} %  ·  {m.get('n_folds', '—')}-fold CV")
                st.markdown(f"**Loss functions:** {', '.join(m['losses']) if m['losses'] else '—'}")
                if m.get("custom_loss"):
                    st.caption(f"Custom: `{m['custom_loss']['name']}` — {m['custom_loss']['expr']}")

            with col_right:
                if m.get("feature_mode") == "Auto":
                    n = len(m["features"])
                    st.markdown(f"**Features:** {n} (auto-selected at training time)")
                else:
                    st.markdown(f"**Features ({len(m['features'])}):**")
                    badges = "  ".join(f"`{f}`" for f in m["features"])
                    st.markdown(badges)

            with col_del:
                if st.button("🗑", key=f"del_{i}", help="Remove this model"):
                    st.session_state.model_list.pop(i)
                    st.rerun()

    st.divider()

    col_run, col_clear = st.columns([3, 1])
    with col_run:
        st.button(
            "▶  Run all models",
            type="primary",
            use_container_width=True,
            disabled=True,
            help="Execution coming soon.",
        )
    with col_clear:
        if st.button("Clear all", use_container_width=True):
            st.session_state.model_list = []
            st.rerun()


def _render_loss_reference() -> None:
    """Compact reference table for built-in loss functions."""
    with st.expander("Loss function reference", expanded=False):
        cols = st.columns(3)
        for idx, (name, formula) in enumerate(LOSS_DESCRIPTIONS.items()):
            with cols[idx % 3]:
                st.markdown(f"**{name}**")
                st.latex(formula)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render() -> None:
    _init_state()

    st.title("Model Specification")
    st.caption("Build a list of model configurations to train and compare.")

    st.divider()

    _render_builder()

    st.divider()

    _render_model_list()

    st.divider()

    _render_loss_reference()
