import streamlit as st

from back_end.service import available_model_types, available_stocks, start_run_from_ui

# ---------------------------------------------------------------------------
# Static reference data (dummy for now)
# ---------------------------------------------------------------------------

FALLBACK_MODELS = [
    "XGBoost",
    "Random Forest",
    "HAR-RV",
    "GARCH(1,1)",
    "Linear Regression",
    "Ridge Regression",
    "LASSO",
    "Decision Tree",
]

AVAILABLE_MODELS = available_model_types() or FALLBACK_MODELS

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
    "XGBoost":            ["Lagged RV (1-step)", "Lagged RV (5-step)", "Rolling RV (60 s)", "Bid-Ask Spread", "Depth Ratio (Level 1)", "Time of Day"],
    "Random Forest":      ["Lagged RV (1-step)", "Lagged RV (5-step)", "Rolling RV (60 s)", "Bid-Ask Spread", "Depth Ratio (Level 1)", "Time of Day"],
    "HAR-RV":             ["Rolling RV (5 s)", "Rolling RV (300 s)", "Lagged RV (1-step)", "Lagged RV (5-step)"],
    "GARCH(1,1)":         ["Mid-Price Change"],
    "Linear Regression":  ["Lagged RV (1-step)", "Lagged RV (5-step)", "Rolling RV (60 s)"],
    "Ridge Regression":   ["Lagged RV (1-step)", "Lagged RV (5-step)", "Rolling RV (60 s)", "Bid-Ask Spread"],
    "LASSO":              ["Lagged RV (1-step)", "Lagged RV (5-step)", "Rolling RV (60 s)", "Bid-Ask Spread", "Order Book Imbalance"],
    "Decision Tree":      ["Lagged RV (1-step)", "Lagged RV (5-step)", "Rolling RV (60 s)", "Depth Ratio (Level 1)", "Time of Day"],
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
    else:
        st.session_state.model_list = [
            {
                key: value
                for key, value in model.items()
                if key != "name"
            }
            for model in st.session_state.model_list
            if model.get("type") in AVAILABLE_MODELS
        ]
    if "builder_feature_mode" not in st.session_state:
        st.session_state.builder_feature_mode = "Auto"
    if "run_status" not in st.session_state:
        st.session_state.run_status = None


def _model_type_icon(model_type: str) -> str:
    icons = {
        "XGBoost": "🌲",
        "Random Forest": "🌳",
        "HAR-RV": "📐",
        "GARCH(1,1)": "📊",
        "Linear Regression": "📏",
        "Ridge Regression": "📉",
        "LASSO": "✂️",
        "Decision Tree": "🌿",
    }
    return icons.get(model_type, "⚙️")


def _model_display_name(models: list[dict], index: int) -> str:
    model_type = models[index]["type"]
    total_of_type = sum(1 for model in models if model.get("type") == model_type)
    if total_of_type == 1:
        return model_type
    occurrence = sum(1 for model in models[: index + 1] if model.get("type") == model_type)
    return f"{model_type} #{occurrence}"


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

    # ---- Model type ----
    model_type = st.selectbox("Model architecture", AVAILABLE_MODELS, key=f"mt_{v}")

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
            default=AUTO_FEATURES.get(model_type, ALL_FEATURES[:5]),
            key=f"sf_{v}",
        )
        if not selected_features:
            st.warning("Select at least one feature.")

    st.divider()

    # ---- Cross-validation ----
    st.markdown("**Cross-validation**")
    st.caption("Each run uses 5 shuffled folds over time windows: 80% training and 20% held out per fold.")

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

    if st.button("➕  Add model to list", type="primary", width="stretch"):
        if feature_mode == "Manual" and not selected_features:
            st.error("Cannot add model: no features selected.")
            return

        losses = list(selected_losses)
        if use_custom and custom_loss_name:
            losses.append(f"{custom_loss_name} (custom)")

        entry = {
            "type": model_type,
            "feature_mode": feature_mode,
            "features": selected_features,
            "pred_seconds": 30,
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
        with st.expander(f"{icon}  **{_model_display_name(models, i)}**", expanded=True):
            col_left, col_right, col_del = st.columns([3, 3, 1])

            with col_left:
                st.markdown("**Cross-validation:** 5 folds  ·  80 / 20 % per fold")
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

    stock_options = available_stocks()
    st.subheader("Run Scope")
    st.caption(
        f"Runs use all {len(stock_options)} stock parquet files and all available time windows per stock."
        if stock_options
        else "No stock parquet files were found."
    )

    col_run, col_clear = st.columns([3, 1])
    with col_run:
        if st.button(
            "▶  Run all models",
            type="primary",
            width="stretch",
            disabled=not stock_options,
            help="Train and evaluate the configured models on every real stock parquet file.",
        ):
            try:
                with st.spinner("Running backend pipeline on real parquet data..."):
                    status = start_run_from_ui(models)
                st.session_state.run_status = status
                st.session_state.selected_run_id = status["run_id"]
                st.session_state.selected_stock = stock_options[0]
                st.success(
                    f"Run {status['run_id']} completed: "
                    f"{status.get('n_predictions', 0):,} predictions across "
                    f"{status.get('n_stocks', len(stock_options))} stock(s)."
                )
                if status.get("feature_source") == "cache":
                    st.caption(f"Feature cache: {status.get('feature_cache_status')}")
                elif status.get("feature_cache_status"):
                    st.warning(f"Feature cache not used: {status.get('feature_cache_status')}")
                if status.get("unsupported_model_types"):
                    st.warning("Some requested models were skipped: " + "; ".join(status["unsupported_model_types"]))
            except Exception as exc:
                st.session_state.run_status = None
                st.error(f"Run failed: {exc}")
    with col_clear:
        if st.button("Clear all", width="stretch"):
            st.session_state.model_list = []
            st.rerun()

    status = st.session_state.get("run_status")
    if status:
        st.caption(
            f"Latest run: {status['run_id']} · "
            f"{status.get('n_feature_rows', 0):,} feature rows · "
            f"{status.get('n_predictions', 0):,} predictions · "
            f"features: {status.get('feature_source', 'unknown')}"
        )


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
