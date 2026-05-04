"""
probe.py — hardcoded probe pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PCA_COMPONENTS = 64
LOGREG_C = 0.1
LOGREG_SEED = 42


class HallucinationProbe(nn.Module):
    """Simple fixed probe: StandardScaler -> PCA -> LogisticRegression."""

    def __init__(self) -> None:
        super().__init__()
        self._pipeline: Pipeline | None = None
        self._threshold: float = 0.5

    @staticmethod
    def _as_2d_float(X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _best_threshold(probs: np.ndarray, y: np.ndarray) -> float:
        candidates = np.linspace(0.10, 0.90, 81)
        best_threshold = 0.5
        best_acc = -1.0
        for threshold in candidates:
            acc = accuracy_score(y, (probs >= threshold).astype(int))
            if acc > best_acc + 1e-12:
                best_acc = acc
                best_threshold = float(threshold)
            elif np.isclose(acc, best_acc) and abs(threshold - 0.5) < abs(best_threshold - 0.5):
                best_threshold = float(threshold)
        return best_threshold

    def _build_pipeline(self, X: np.ndarray) -> Pipeline:
        max_components = min(X.shape[0] - 1, X.shape[1], PCA_COMPONENTS)
        steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
        if max_components >= 2:
            steps.append(("pca", PCA(n_components=int(max_components), svd_solver="auto", random_state=LOGREG_SEED)))
        steps.append(
            (
                "logreg",
                LogisticRegression(
                    C=LOGREG_C,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=LOGREG_SEED,
                ),
            )
        )
        return Pipeline(steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x.detach().cpu().numpy())[:, 1]
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(probs / (1.0 - probs))
        return torch.from_numpy(logits).to(dtype=x.dtype, device=x.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_arr = self._as_2d_float(X)
        y_arr = np.asarray(y).astype(int)
        self._pipeline = self._build_pipeline(X_arr)
        self._pipeline.fit(X_arr, y_arr)
        self._threshold = 0.5
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        probs = self.predict_proba(self._as_2d_float(X_val))[:, 1]
        self._threshold = self._best_threshold(probs, np.asarray(y_val).astype(int))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Probe is not fitted yet. Call fit() before predict_proba().")

        X_arr = self._as_2d_float(X)
        probs = self._pipeline.predict_proba(X_arr)
        classes = list(self._pipeline.classes_)
        pos_idx = classes.index(1)
        prob_pos = np.clip(probs[:, pos_idx], 0.0, 1.0)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
