from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


ROOT_DIR = Path(__file__).resolve().parents[1]
INDIVIDUAL_PARQUET_DIR = ROOT_DIR / "individual_book_train_parquet"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FEATURE_CACHE_DIR = ARTIFACTS_DIR / "feature_cache"
FEATURE_CACHE_PATH = FEATURE_CACHE_DIR / "features_all_stocks.parquet"
DATA_DIR = str(INDIVIDUAL_PARQUET_DIR)
CLUSTER_CSV = str(ROOT_DIR / "plots" / "recluster" / "stock_cluster_assignments_FINAL.csv")
PLOTS_DIR = str(ROOT_DIR / "plots")

PIPELINE_VERSION = "2026-05-10.2"

RETURN_WINDOWS = [
    5,
    10,
    20,
    30,
    45,
    60,
    90,
    120,
    180,
    240,
    300,
    330,
    350,
    380,
    400,
    420,
    450,
    500,
]
ACF_WINDOWS = [30, 60, 120, 300]
BOOK_WINDOWS = [30, 60, 120, 300]
EWMA_LAMBDAS = [0.80, 0.90, 0.94, 0.97]
EPS = 1e-9


FEATURE_LABEL_MAP = {
    "WAP Returns": ["last_return", "abs_last_return", "mean_abs_ret_30"],
    "Bid-Ask Spread": ["mean_spread_30", "std_spread_30", "mean_spread_300"],
    "Order Book Imbalance": ["mean_imbalance_30", "book_pressure_30"],
    "Rolling RV (5 s)": ["RV_5", "vol_5"],
    "Rolling RV (30 s)": ["RV_30", "vol_30"],
    "Rolling RV (60 s)": ["RV_60", "vol_60"],
    "Rolling RV (300 s)": ["RV_300", "vol_300"],
    "Depth Ratio (Level 1)": ["depth_ratio_30"],
    "Depth Ratio (Levels 1-3)": ["depth_ratio_300", "mean_depth_300"],
    "Volume Imbalance": ["mean_imbalance_300", "std_imbalance_300"],
    "Mid-Price Change": ["last_return", "mean_ret_30"],
    "Time of Day": ["seconds_elapsed", "cutoff"],
    "Day of Week": [],
    "Lagged RV (1-step)": ["RV_30", "RV_60"],
    "Lagged RV (5-step)": ["RV_300", "RV_420"],
}

FEATURE_COLS = [
    "rv_first",
    "std_log_return",
    "skew_log_return",
    "kurt_log_return",
    "max_log_return",
    "min_log_return",
    "realized_quarticity",
    "abs_return_mean",
    "max_abs_return",
    "wap_range",
    "wap_std",
    "n_price_changes",
    "n_rows",
    "mean_spread",
    "std_spread",
    "spread_range",
    "mean_rel_spread",
    "std_rel_spread",
    "mean_depth",
    "std_depth",
    "mean_imbalance",
    "std_imbalance",
    "mean_bid_size1",
    "mean_ask_size1",
    "arrival_rate",
    "wap_momentum",
    "acf1",
    "acf5",
    "pacf1",
    "pacf5",
    "rv_first_lag1",
    "rv_first_lag2",
    "log_rv_lag1",
    "log_rv_lag2",
    "log_rv_lag3",
    "log_rv_rolling5",
    "vol_per_depth",
    "vol_x_spread",
    "pressure_x_vol",
]


@dataclass(frozen=True)
class ModelDefinition:
    label: str
    icon: str
    required_module: str | None = None


MODEL_REGISTRY: dict[str, ModelDefinition] = {
    "XGBoost": ModelDefinition(
        label="XGBoost",
        icon="🌲",
        required_module="xgboost",
    ),
    "Random Forest": ModelDefinition(
        label="Random Forest",
        icon="🌳",
    ),
    "HAR-RV": ModelDefinition(
        label="HAR-RV",
        icon="📐",
    ),
    "GARCH(1,1)": ModelDefinition(
        label="GARCH(1,1)",
        icon="📊",
        required_module="arch",
    ),
    "EGARCH(1,1)": ModelDefinition(
        label="EGARCH(1,1)",
        icon="📈",
        required_module="arch",
    ),
    "GJR-GARCH(1,1)": ModelDefinition(
        label="GJR-GARCH(1,1)",
        icon="📊",
        required_module="arch",
    ),
    "Linear Regression": ModelDefinition(
        label="Linear Regression",
        icon="📏",
    ),
    "Ridge Regression": ModelDefinition(
        label="Ridge Regression",
        icon="📉",
    ),
    "LASSO": ModelDefinition(
        label="LASSO",
        icon="✂️",
    ),
    "Decision Tree": ModelDefinition(
        label="Decision Tree",
        icon="🌿",
    ),
}

SUPPORTED_MODEL_TYPES = tuple(MODEL_REGISTRY)

MODEL_TYPE_ALIASES = {
    **{model_type: model_type for model_type in SUPPORTED_MODEL_TYPES},
    "EGARCH": "EGARCH(1,1)",
    "E-GARCH": "EGARCH(1,1)",
    "E-GARCH(1,1)": "EGARCH(1,1)",
    "E GARCH": "EGARCH(1,1)",
    "E GARCH(1,1)": "EGARCH(1,1)",
    "GJR-GARCH": "GJR-GARCH(1,1)",
    "GJR GARCH": "GJR-GARCH(1,1)",
    "GJR GARCH(1,1)": "GJR-GARCH(1,1)",
    "Lasso": "LASSO",
    "lasso": "LASSO",
}

FRONTEND_MODEL_METRICS = tuple(
    {"model": model, "inference_us": None, "rmse": None, "qlike": None}
    for model in SUPPORTED_MODEL_TYPES
)


def model_catalog(model_types: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    selected = SUPPORTED_MODEL_TYPES if model_types is None else model_types
    return [
        {
            "type": model_type,
            "label": definition.label,
            "icon": definition.icon,
        }
        for model_type in selected
        for definition in [MODEL_REGISTRY[model_type]]
    ]


@dataclass(frozen=True)
class DataConfig:
    stocks: tuple[str, ...] = ("stock_0",)
    source_dir: str = str(INDIVIDUAL_PARQUET_DIR)
    max_time_ids_per_stock: int | None = None
    feature_cache_path: str | None = str(FEATURE_CACHE_PATH)
    use_feature_cache: bool = True


@dataclass(frozen=True)
class FeatureConfig:
    forecast_horizon: int = 30
    return_windows: tuple[int, ...] = tuple(RETURN_WINDOWS)
    acf_windows: tuple[int, ...] = tuple(ACF_WINDOWS)
    book_windows: tuple[int, ...] = tuple(BOOK_WINDOWS)
    ewma_lambdas: tuple[float, ...] = tuple(EWMA_LAMBDAS)
    feature_mode: Literal["PCA", "Manual"] = "PCA"
    n_pca_components: int = 12
    manual_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitConfig:
    train_pct: int = 80
    n_folds: int = 5
    shuffle: bool = True
    random_state: int = 42


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    feature_mode: Literal["PCA", "Manual"] | None = None
    n_pca_components: int | None = None
    manual_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class UniverseConfig:
    enabled: bool = True


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    models: tuple[ModelSpec, ...] = (
        ModelSpec(name="HAR-RV", model_type="HAR-RV"),
        ModelSpec(name="Linear Regression", model_type="Linear Regression"),
        ModelSpec(name="Random Forest", model_type="Random Forest"),
    )
    universe: UniverseConfig = field(default_factory=UniverseConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_stock_id(stock: str | int) -> str:
    text = str(stock)
    return text if text.startswith("stock_") else f"stock_{text}"


def _ui_model_type(entry: dict[str, Any]) -> str:
    requested_type = entry.get("type", "Linear Regression")
    return MODEL_TYPE_ALIASES.get(requested_type, requested_type)


def _canonical_model_name(model_type: str, occurrence: int = 1, total: int = 1) -> str:
    return model_type if total <= 1 else f"{model_type} #{occurrence}"


def model_spec_from_ui(entry: dict[str, Any], *, name: str | None = None) -> ModelSpec:
    """Convert one UI model entry.

    UI-supplied custom model names are deprecated and intentionally ignored.
    """
    feature_mode = entry.get("feature_mode", "PCA")
    if feature_mode not in {"PCA", "Manual"}:
        raise ValueError(f"Unsupported feature selection mode: {feature_mode}")
    features = entry.get("features", ())
    n_pca_components = len(features) if feature_mode == "PCA" else None
    manual_features = tuple(features) if feature_mode == "Manual" else ()
    model_type = _ui_model_type(entry)
    return ModelSpec(
        name=name or _canonical_model_name(model_type),
        model_type=model_type,
        parameters=dict(entry.get("parameters") or {}),
        feature_mode=feature_mode,
        n_pca_components=n_pca_components,
        manual_features=manual_features,
    )


def model_specs_from_ui(entries: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> tuple[ModelSpec, ...]:
    model_types = [_ui_model_type(entry) for entry in entries]
    totals = {model_type: model_types.count(model_type) for model_type in set(model_types)}
    seen: dict[str, int] = {}
    specs = []
    for entry, model_type in zip(entries, model_types):
        seen[model_type] = seen.get(model_type, 0) + 1
        specs.append(
            model_spec_from_ui(
                entry,
                name=_canonical_model_name(model_type, seen[model_type], totals[model_type]),
            )
        )
    return tuple(specs)
