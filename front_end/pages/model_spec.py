import streamlit as st

from back_end.service import (
    available_cached_runs,
    available_model_catalog,
    available_stocks,
    load_pca_variance_explained,
    start_run_from_ui,
)
from charts import pca_cumulative_variance_chart, pca_variance_explained_chart

# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------

MODEL_CATALOG = available_model_catalog()
AVAILABLE_MODELS = [model["type"] for model in MODEL_CATALOG]
MODEL_ICONS = {model["type"]: model["icon"] for model in MODEL_CATALOG}
MAX_PCA_COMPONENTS = 30
DEFAULT_GARCH_N_JOBS = 4

MODEL_PARAMETER_DEFAULTS = {
    "XGBoost": {
        "n_estimators": 180,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "Random Forest": {
        "n_estimators": 60,
    },
    "Decision Tree": {
        "max_depth": 6,
        "min_samples_leaf": 20,
    },
    "LASSO": {
        "alpha": 0.01,
        "max_iter": 5000,
    },
    "GARCH(1,1)": {
        "n_jobs": DEFAULT_GARCH_N_JOBS,
    },
    "GJR-GARCH(1,1)": {
        "n_jobs": DEFAULT_GARCH_N_JOBS,
    },
}

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
    "Lagged RV (1-step)",
    "Lagged RV (5-step)",
]

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
                if key in {"type", "feature_mode", "features", "pred_seconds", "parameters"}
            }
            for model in st.session_state.model_list
            if model.get("type") in AVAILABLE_MODELS
            and model.get("feature_mode", "PCA") in {"PCA", "Manual"}
        ]
    if "run_status" not in st.session_state:
        st.session_state.run_status = None


def _model_type_icon(model_type: str) -> str:
    return MODEL_ICONS.get(model_type, "⚙️")


def _model_display_name(models: list[dict], index: int) -> str:
    model_type = models[index]["type"]
    total_of_type = sum(1 for model in models if model.get("type") == model_type)
    if total_of_type == 1:
        return model_type
    occurrence = sum(1 for model in models[: index + 1] if model.get("type") == model_type)
    return f"{model_type} #{occurrence}"


def _model_entry(
    model_type: str,
    feature_mode: str,
    selected_features: list[str] | list[int],
    parameters: dict,
) -> dict:
    model_parameters = _model_parameters_for_type(model_type, parameters)
    return {
        "type": model_type,
        "feature_mode": feature_mode,
        "features": list(selected_features),
        "pred_seconds": 30,
        "parameters": model_parameters,
    }


def _model_parameters_for_type(model_type: str, parameters: dict) -> dict:
    defaults = MODEL_PARAMETER_DEFAULTS.get(model_type, {})
    return {
        key: parameters.get(key, default)
        for key, default in defaults.items()
    }


def _format_parameter_value(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _parameter_summary(parameters: dict | None) -> str:
    if not parameters:
        return "Default backend settings"
    return ", ".join(
        f"{name}={_format_parameter_value(value)}"
        for name, value in parameters.items()
    )


def _render_parameter_controls(model_type: str, parameters: dict, key_prefix: str) -> dict:
    defaults = MODEL_PARAMETER_DEFAULTS.get(model_type)
    if not defaults:
        return {}

    values = _model_parameters_for_type(model_type, parameters)
    st.markdown("**Hyperparameters**")
    if model_type == "XGBoost":
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            values["n_estimators"] = st.number_input(
                "Trees",
                min_value=10,
                max_value=1000,
                value=int(values["n_estimators"]),
                step=10,
                key=f"{key_prefix}_xgb_ne",
            )
        with col_b:
            values["learning_rate"] = st.number_input(
                "Learning rate",
                min_value=0.001,
                max_value=1.0,
                value=float(values["learning_rate"]),
                step=0.01,
                format="%.3f",
                key=f"{key_prefix}_xgb_lr",
            )
        with col_c:
            values["max_depth"] = st.number_input(
                "Max depth",
                min_value=1,
                max_value=20,
                value=int(values["max_depth"]),
                step=1,
                key=f"{key_prefix}_xgb_md",
            )
        col_d, col_e = st.columns(2)
        with col_d:
            values["subsample"] = st.slider(
                "Subsample",
                min_value=0.1,
                max_value=1.0,
                value=float(values["subsample"]),
                step=0.05,
                key=f"{key_prefix}_xgb_ss",
            )
        with col_e:
            values["colsample_bytree"] = st.slider(
                "Column sample by tree",
                min_value=0.1,
                max_value=1.0,
                value=float(values["colsample_bytree"]),
                step=0.05,
                key=f"{key_prefix}_xgb_cs",
            )
    elif model_type == "Random Forest":
        values["n_estimators"] = st.number_input(
            "Trees",
            min_value=10,
            max_value=1000,
            value=int(values["n_estimators"]),
            step=10,
            key=f"{key_prefix}_rf_ne",
        )
    elif model_type == "Decision Tree":
        col_a, col_b = st.columns(2)
        with col_a:
            values["max_depth"] = st.number_input(
                "Max depth",
                min_value=1,
                max_value=50,
                value=int(values["max_depth"]),
                step=1,
                key=f"{key_prefix}_dt_md",
            )
        with col_b:
            values["min_samples_leaf"] = st.number_input(
                "Minimum samples per leaf",
                min_value=1,
                max_value=500,
                value=int(values["min_samples_leaf"]),
                step=1,
                key=f"{key_prefix}_dt_msl",
            )
    elif model_type == "LASSO":
        col_a, col_b = st.columns(2)
        with col_a:
            values["alpha"] = st.number_input(
                "Alpha",
                min_value=0.0001,
                max_value=10.0,
                value=float(values["alpha"]),
                step=0.01,
                format="%.4f",
                key=f"{key_prefix}_lasso_alpha",
            )
        with col_b:
            values["max_iter"] = st.number_input(
                "Maximum iterations",
                min_value=100,
                max_value=50000,
                value=int(values["max_iter"]),
                step=100,
                key=f"{key_prefix}_lasso_iter",
            )
    elif "GARCH" in model_type:
        values["n_jobs"] = st.slider(
            "Parallel workers",
            min_value=1,
            max_value=8,
            value=int(values["n_jobs"]),
            step=1,
            key=f"{key_prefix}_gj",
            help="Runs GARCH fits across stocks in separate worker processes.",
        )

    return values


def _cached_run_label(run: dict) -> str:
    n_predictions = int(run.get("n_predictions", 0))
    n_stocks = int(run.get("n_stocks", 0))
    return f"{run['run_id']} · {n_predictions:,} predictions · {n_stocks} stock(s)"


# ---------------------------------------------------------------------------
# Sub-sections
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_pca_variance(stocks: tuple[str, ...]):
    return load_pca_variance_explained(MAX_PCA_COMPONENTS, stocks)


def _render_builder() -> None:
    """Configures and adds a model to the list.

    Uses plain widgets (no st.form) so the feature-mode radio triggers an
    immediate rerender without waiting for a submit button.  A 'form version'
    counter in session state is incremented on submit to reset all widget keys.
    """
    st.subheader("Configure a Model")

    if not AVAILABLE_MODELS:
        st.error("No backend models are currently available.")
        return

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
        ["PCA", "Manual"],
        index=0,
        horizontal=True,
        key=f"fm_{v}",
        help="PCA: compress all usable features into components. Manual: pick feature groups by name.",
    )

    if feature_mode == "PCA":
        n_features = st.slider(
            "Number of PCA components",
            min_value=1,
            max_value=MAX_PCA_COMPONENTS,
            value=15,
            step=1,
            key=f"nf_{v}",
            help="All usable raw features are standardised, then compressed into N principal components.",
        )
        selected_features = list(range(n_features))
        with st.spinner("Computing PCA variance explained..."):
            pca_variance = _cached_pca_variance(tuple(available_stocks()))
        st.plotly_chart(
            pca_variance_explained_chart(pca_variance, n_features),
            width="stretch",
        )
        st.plotly_chart(
            pca_cumulative_variance_chart(pca_variance, n_features),
            width="stretch",
        )
    else:
        selected_features = st.multiselect(
            "Select features",
            options=ALL_FEATURES,
            default=[],
            key=f"sf_{v}",
        )
        if not selected_features:
            st.warning("Select at least one feature.")

    st.divider()

    # ---- Loss functions ----
    st.markdown("**Loss function(s)**")
    st.caption(f"Using all built-in loss functions: {', '.join(LOSS_FUNCTIONS)}")

    st.divider()

    col_add, col_add_all = st.columns([3, 1])
    with col_add:
        if st.button("➕  Add model to list", type="primary", width="stretch"):
            if feature_mode == "Manual" and not selected_features:
                st.error("Cannot add model: no features selected.")
                return

            entry = _model_entry(model_type, feature_mode, selected_features, {})
            st.session_state.model_list.append(entry)
            st.session_state.builder_v = v + 1  # reset all widget values
            st.rerun()
    with col_add_all:
        if st.button(
            "Add all models",
            width="stretch",
            help="Add every available backend model that is not already configured.",
        ):
            if feature_mode == "Manual" and not selected_features:
                st.error("Cannot add models: no features selected.")
                return

            configured_types = {model.get("type") for model in st.session_state.model_list}
            entries = [
                _model_entry(available_model, feature_mode, selected_features, {})
                for available_model in AVAILABLE_MODELS
                if available_model not in configured_types
            ]
            if not entries:
                st.info("All available models are already configured.")
                return

            st.session_state.model_list.extend(entries)
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
                st.markdown(f"**Loss functions:** {', '.join(LOSS_FUNCTIONS)}")

            with col_right:
                if m.get("feature_mode") == "PCA":
                    n = len(m["features"])
                    st.markdown(f"**Features:** {n} PCA component{'s' if n != 1 else ''}")
                else:
                    st.markdown(f"**Features ({len(m['features'])}):**")
                    badges = "  ".join(f"`{f}`" for f in m["features"])
                    st.markdown(badges)

            with col_del:
                if st.button("🗑", key=f"del_{i}", help="Remove this model"):
                    st.session_state.model_list.pop(i)
                    st.rerun()

            if m["type"] in MODEL_PARAMETER_DEFAULTS:
                m["parameters"] = _render_parameter_controls(
                    m["type"],
                    m.get("parameters") or {},
                    f"model_{i}",
                )
                st.markdown(f"**Hyperparameters:** {_parameter_summary(m['parameters'])}")

    st.divider()

    stock_options = available_stocks()
    st.subheader("Run Scope")
    st.caption(
        f"Runs use all {len(stock_options)} stock CSV files and all available time windows per stock."
        if stock_options
        else "No stock CSV files were found."
    )

    col_run, col_clear = st.columns([3, 1])
    with col_run:
        if st.button(
            "▶  Run all models",
            type="primary",
            width="stretch",
            disabled=not stock_options,
            help="Train and evaluate the configured models on every real stock CSV file.",
        ):
            try:
                with st.spinner("Running backend pipeline on real CSV data..."):
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
            except Exception as exc:
                st.session_state.run_status = None
                st.error(f"Run failed: {exc}")

        cached_runs = available_cached_runs()
        if cached_runs:
            run_ids = [run["run_id"] for run in cached_runs]
            saved_run = st.session_state.get("selected_run_id")
            selected_index = run_ids.index(saved_run) if saved_run in run_ids else 0
            labels = {run["run_id"]: _cached_run_label(run) for run in cached_runs}
            selected_run_id = st.selectbox(
                "Display cached run",
                options=run_ids,
                index=selected_index,
                format_func=labels.get,
                key="selected_run_id",
                help="Choose which cached backend run feeds the Individual Stock and Universe tabs.",
            )
            st.session_state.run_status = next(
                run for run in cached_runs if run["run_id"] == selected_run_id
            )
        else:
            st.caption("No cached runs available yet.")
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
