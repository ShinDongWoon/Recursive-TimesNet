from __future__ import annotations

from typing import Dict, Sequence, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.time_features import build_time_features


def _coerce_datetime_index(
    index: Optional[pd.DatetimeIndex | np.ndarray], expected_len: int
) -> Optional[pd.DatetimeIndex]:
    if index is None:
        return None
    if isinstance(index, pd.DatetimeIndex):
        idx = index
    else:
        idx = pd.to_datetime(np.asarray(index))
    if len(idx) != expected_len:
        raise ValueError(
            "time_index length must match the first dimension of wide_values"
        )
    return idx


class SlidingWindowDataset(Dataset):
    """Sliding-window access to wide-format time series arrays.

    The dataset returns tuples structured as ``(x, y, mask, x_mark, y_mark,``
    ``series_static, series_ids)`` where temporal mark tensors may be ``None``
    when time feature construction is disabled. Downstream consumers should
    therefore guard against missing marks before use.
    """
    def __init__(
        self,
        wide_values: np.ndarray,  # [T, N]
        input_len: int,
        pred_len: int,
        mode: str,  # "direct"|"recursive"
        recursive_pred_len: int | None = None,
        augment: Dict | None = None,
        valid_mask: np.ndarray | None = None,  # [T, N]
        series_static: np.ndarray | None = None,
        series_ids: Sequence[int] | np.ndarray | None = None,
        time_index: pd.DatetimeIndex | np.ndarray | None = None,
        time_features: np.ndarray | None = None,
        time_feature_config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        assert mode in ("direct", "recursive")
        self.X = wide_values.astype(np.float32)
        if valid_mask is not None and valid_mask.shape != self.X.shape:
            raise ValueError(
                "valid_mask must match wide_values shape"
            )
        if valid_mask is None:
            self.M = np.ones_like(self.X, dtype=np.float32)
        else:
            self.M = valid_mask.astype(np.float32)
        self.T, self.N = self.X.shape
        if self.N <= 0:
            raise ValueError("wide_values must contain at least one series column")
        self.L = int(input_len)
        if mode == "direct":
            self.H = int(pred_len)
        else:
            # In recursive mode training windows normally have one-step targets.
            # ``recursive_pred_len`` can override this to produce multi-step
            # targets (e.g. for validation).
            self.H = int(recursive_pred_len if recursive_pred_len is not None else 1)
        self.mode = mode
        augment = augment or {}
        self.add_noise_std = float(augment.get("add_noise_std", 0.0))
        self.time_shift = int(augment.get("time_shift", 0))
        # Indices where a full (L,H) window fits. ``self.H`` is the mode-specific
        # output length, so the last valid start index is ``T - L - H``.
        self.idxs = np.arange(self.T - self.L - self.H + 1)
        # In recursive mode ``self.H`` may be 1 (training) or >1 (validation).
        self._X_tensor = torch.from_numpy(self.X)
        self._M_tensor = torch.from_numpy(self.M)
        if torch.cuda.is_available():
            self._X_tensor = self._X_tensor.pin_memory()
            self._M_tensor = self._M_tensor.pin_memory()

        self.time_feature_dim: int = 0
        self.time_feature_config = dict(time_feature_config or {})
        idx = _coerce_datetime_index(time_index, self.T)
        marks_tensor: torch.Tensor | None = None
        if time_features is not None:
            feats = np.asarray(time_features)
            if feats.shape[0] != self.T:
                raise ValueError(
                    "time_features must align with the temporal dimension of wide_values"
                )
            if feats.ndim == 1:
                feats = feats.reshape(-1, 1)
            if feats.ndim != 2:
                raise ValueError(
                    "time_features must be a 2D array of shape [T, F]"
                )
            self.time_feature_dim = int(feats.shape[1])
            if self.time_feature_dim > 0:
                marks_tensor = torch.from_numpy(
                    feats.astype(np.float32, copy=False)
                )
        elif idx is not None and self.time_feature_config.get("enabled", False):
            feats = build_time_features(idx, self.time_feature_config)
            if feats.ndim == 1:
                feats = feats.reshape(-1, 1)
            if feats.shape[0] != self.T:
                raise ValueError(
                    "Computed time features must align with wide_values"
                )
            self.time_feature_dim = int(feats.shape[1]) if feats.ndim == 2 else 0
            if self.time_feature_dim > 0:
                marks_tensor = torch.from_numpy(feats)
        elif time_feature_config and time_feature_config.get("enabled", False):
            raise ValueError(
                "time_feature_config.enabled is True but no time_index or precomputed time_features were provided"
            )
        if marks_tensor is not None:
            if torch.cuda.is_available():
                self.time_marks = marks_tensor.pin_memory()
            else:
                self.time_marks = marks_tensor
        else:
            self.time_marks = None
        if isinstance(self.time_feature_dim, int) and self.time_feature_dim <= 0:
            self.time_marks = None

        if self.time_marks is not None:
            self.time_feature_dim = int(self.time_marks.size(-1))

        self._time_freq = idx.freqstr if idx is not None else None
        if torch.cuda.is_available():
            self._empty_time_mark = torch.empty(0, dtype=torch.float32).pin_memory()
        else:
            self._empty_time_mark = torch.empty(0, dtype=torch.float32)

        if series_static is not None:
            static_arr = np.asarray(series_static, dtype=np.float32)
            if static_arr.ndim == 1:
                static_arr = static_arr.reshape(-1, 1)
            if static_arr.shape[0] != self.N:
                raise ValueError(
                    "series_static must have shape [num_series, num_features]"
                )
            tensor = torch.from_numpy(static_arr)
            if torch.cuda.is_available():
                tensor = tensor.pin_memory()
            self.series_static = tensor
        else:
            self.series_static = None

        if series_ids is not None:
            ids_arr = np.asarray(series_ids)
            if ids_arr.ndim != 1:
                raise ValueError("series_ids must be a 1D sequence")
            if ids_arr.shape[0] != self.N:
                raise ValueError("series_ids length must match number of series")
            ids_tensor = torch.from_numpy(ids_arr.astype(np.int64, copy=False))
            if torch.cuda.is_available():
                ids_tensor = ids_tensor.pin_memory()
            self.series_ids = ids_tensor
        else:
            self.series_ids = None

        self._windows_per_series = int(len(self.idxs))

    def __len__(self) -> int:
        return int(self._windows_per_series * self.N)

    def __getitem__(self, idx: int) -> tuple[object, ...]:
        if self._windows_per_series <= 0:
            raise IndexError("SlidingWindowDataset is empty")
        window_idx = int(idx // self.N)
        series_idx = int(idx % self.N)
        if window_idx >= self._windows_per_series:
            raise IndexError("index out of range for sliding windows")
        s = int(self.idxs[window_idx])
        if self.time_shift > 0:
            delta = np.random.randint(-self.time_shift, self.time_shift + 1)
            s = int(np.clip(s + delta, 0, self.T - self.L - self.H))
        e = s + self.L
        # ``narrow`` avoids extra tensor allocations while keeping slices
        # compatible with the default collation logic. Clones are only
        # materialized when augmentations mutate the data.
        x_tensor = self._X_tensor.narrow(0, s, self.L).narrow(1, series_idx, 1)
        if self.add_noise_std > 0:
            x_tensor = x_tensor.clone()
            noise = torch.randn_like(x_tensor) * self.add_noise_std
            x_tensor = x_tensor + noise
        y_tensor = self._X_tensor.narrow(0, e, self.H).narrow(1, series_idx, 1)
        mask_tensor = self._M_tensor.narrow(0, e, self.H).narrow(1, series_idx, 1)
        if self.time_marks is not None:
            x_mark = self.time_marks.narrow(0, s, self.L)
            y_mark = self.time_marks.narrow(0, e, self.H)
        else:
            x_mark = self._empty_time_mark
            y_mark = self._empty_time_mark
        items: list[object] = [x_tensor, y_tensor, mask_tensor, x_mark, y_mark]
        if self.series_static is not None:
            static_slice = self.series_static.narrow(0, series_idx, 1)
            items.append(static_slice)
        if self.series_ids is not None:
            id_slice = self.series_ids.narrow(0, series_idx, 1)
            items.append(id_slice)
        return tuple(items)

    @staticmethod
    def has_time_marks(batch: Sequence[object]) -> bool:
        if not isinstance(batch, (list, tuple)):
            return False
        if len(batch) < 5:
            return False
        return batch[3] is not None and batch[4] is not None

    @property
    def time_frequency(self) -> Optional[str]:
        return self._time_freq
