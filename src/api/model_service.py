"""Loads the Delhi bus delay model once at process startup and exposes it
to the API layer.

Configuration (environment variables):
  MODEL_ALIAS   Registry alias to load when MODEL_PATH is not set.
                Defaults to "production". Passed straight to
                src.models.registry.load_model().
  MODEL_PATH    Optional path to a local .joblib file. When set, the model
                is loaded directly from disk and the W&B registry is never
                contacted - useful for Docker images that bake the model in.
  WANDB_API_KEY / WANDB_BASE_URL
                Read directly by the wandb client when MODEL_PATH is not
                set and a registry alias must be resolved remotely; no
                extra handling is needed here.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.registry import ProductionModel, load_model

logger = logging.getLogger("model_service")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_REPORT_PATH = PROJECT_ROOT / "reports" / "model_registry.json"


class ModelService:
    """Holds the single loaded model instance for the lifetime of the app."""

    def __init__(self) -> None:
        self._model: ProductionModel | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model: ProductionModel | None = None) -> None:
        """Load the configured model. Pass `model` directly in tests to
        inject a fake model without touching W&B or the filesystem."""
        with self._lock:
            if model is not None:
                self._model = model
                return

            model_path = os.environ.get("MODEL_PATH")
            alias = os.environ.get("MODEL_ALIAS", "production")
            source = model_path if model_path else alias
            logger.info("model_service: loading model source=%s", source)
            self._model = load_model(source)

    def unload(self) -> None:
        with self._lock:
            self._model = None

    def predict(self, df: pd.DataFrame) -> Any:
        if self._model is None:
            raise RuntimeError("Model is not loaded")
        ordered = df[self._model.expected_feature_columns]
        return self._model.predict(ordered)

    def _registry_git_commit(self) -> str | None:
        if not REGISTRY_REPORT_PATH.exists():
            return None
        try:
            record = json.loads(REGISTRY_REPORT_PATH.read_text())
            return record.get("metadata", {}).get("git_commit")
        except Exception:
            return None

    def info(self) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Model is not loaded")
        return {
            "alias": self._model.alias,
            "version": self._model.version,
            "model_name": self._model.model_name,
            "git_commit": self._registry_git_commit(),
            "feature_columns": self._model.expected_feature_columns,
            "categorical_features": self._model.categorical_features,
            "numeric_features": self._model.numeric_features,
        }


model_service = ModelService()
