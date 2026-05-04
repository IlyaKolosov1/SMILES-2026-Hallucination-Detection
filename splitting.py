"""
splitting.py — hardcoded split pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    del df, test_size

    y = np.asarray(y).astype(int)
    idx = np.arange(len(y), dtype=int)

    outer_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold_idx, (idx_train_val, idx_test) in enumerate(outer_cv.split(idx, y)):
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=val_size,
            random_state=random_state + fold_idx,
            stratify=y[idx_train_val],
        )

        splits.append(
            (
                np.sort(idx_train.astype(int)),
                np.sort(idx_val.astype(int)),
                np.sort(idx_test.astype(int)),
            )
        )

    return splits
