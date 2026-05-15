from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ARTIFACTS_DIR


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_dir(run_id: str) -> Path:
    path = ARTIFACTS_DIR / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_run_artifacts(
    run_id: str,
    config: Any,
    status: dict[str, Any],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    universe_summary: pd.DataFrame,
    similarity: pd.DataFrame,
) -> Path:
    directory = run_dir(run_id)
    write_json(directory / "config.json", config.to_dict() if hasattr(config, "to_dict") else config)
    write_json(directory / "status.json", status)
    predictions.to_parquet(directory / "predictions.parquet", index=False)
    metrics.to_parquet(directory / "metrics.parquet", index=False)
    feature_importance.to_parquet(directory / "feature_importance.parquet", index=False)
    universe_summary.to_parquet(directory / "universe_summary.parquet", index=False)
    similarity.to_parquet(directory / "stock_similarity.parquet", index=True)
    return directory


def list_runs() -> list[dict[str, Any]]:
    root = ARTIFACTS_DIR / "runs"
    if not root.exists():
        return []
    rows = []
    for directory in sorted(root.iterdir(), reverse=True):
        if not directory.is_dir():
            continue
        status_path = directory / "status.json"
        status = read_json(status_path) if status_path.exists() else {}
        rows.append({"run_id": directory.name, **status})
    return rows


def load_run_frame(run_id: str, name: str) -> pd.DataFrame:
    path = run_dir(run_id) / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_run_config(run_id: str) -> dict[str, Any]:
    path = run_dir(run_id) / "config.json"
    return read_json(path) if path.exists() else {}


def latest_run_id() -> str | None:
    runs = list_runs()
    return runs[0]["run_id"] if runs else None
