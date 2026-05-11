from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


ROOT_DIR = Path(__file__).resolve().parents[1]
INDIVIDUAL_PARQUET_DIR = ROOT_DIR / "individual_book_train_parquet"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FEATURE_CACHE_DIR = ARTIFACTS_DIR / "feature_cache"
FEATURE_CACHE_PATH = FEATURE_CACHE_DIR / "features_all_stocks.parquet"

PIPELINE_VERSION = "2026-05-10.2"

FRONTEND_MODEL_ORDER = (
    "XGBoost",
    "Random Forest",
    "HAR-RV",
    "GARCH(1,1)",
    "Linear Regression",
    "Ridge Regression",
    "LASSO",
    "Decision Tree",
)

FRONTEND_MODEL_METRICS = tuple(
    {"model": model, "inference_us": None, "rmse": None, "qlike": None, "pred_target": None}
    for model in FRONTEND_MODEL_ORDER
)

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

MODEL_TYPE_ALIASES = {
    "Linear Regression": "Linear Regression",
    "Ridge Regression": "Ridge Regression",
    "LASSO": "LASSO",
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
    "HAR-RV": "HAR-RV",
    "GARCH(1,1)": "GARCH(1,1)",
}


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
    feature_mode: Literal["Auto", "Manual"] = "Auto"
    n_auto_features: int = 12
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
    feature_mode: Literal["Auto", "Manual"] | None = None
    n_auto_features: int | None = None
    manual_features: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ("RMSE", "QLIKE")


@dataclass(frozen=True)
class UniverseConfig:
    enabled: bool = True
    n_clusters: int = 4
    similarity_metric: str = "realized_variance_correlation"


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
    feature_mode = entry.get("feature_mode", "Auto")
    features = entry.get("features", ())
    n_auto_features = len(features) if feature_mode == "Auto" else None
    manual_features = tuple(features) if feature_mode == "Manual" else ()
    model_type = _ui_model_type(entry)
    return ModelSpec(
        name=name or _canonical_model_name(model_type),
        model_type=model_type,
        feature_mode=feature_mode,
        n_auto_features=n_auto_features,
        manual_features=manual_features,
        metrics=tuple(entry.get("losses") or ("RMSE", "QLIKE")),
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
