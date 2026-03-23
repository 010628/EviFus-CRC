"""
Dataset utilities for EviFus-CRC.

This module is intentionally CSV-driven:
- explicit feature paths in the CSV are used whenever available;
- fallback feature roots are used only when the CSV does not provide paths;
- missing modalities are represented by zero vectors plus a presence mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def zscore_1d(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-sample z-score normalization for pathology embeddings."""
    mean = float(np.mean(x))
    std = float(np.std(x)) + eps
    return (x - mean) / std


def reduce_to_1d_feature(x: np.ndarray) -> np.ndarray:
    """Collapse an arbitrary numpy feature tensor to a 1D embedding."""
    x = np.asarray(x)
    x = np.squeeze(x)

    if x.ndim == 0:
        return np.array([float(x)], dtype=np.float32)

    while x.ndim > 1:
        x = x.mean(axis=0)

    return x.astype(np.float32)


def infer_time_bins_from_training(
    durations: Sequence[float],
    num_bins: int,
) -> List[float]:
    durations = np.asarray(durations, dtype=np.float32)
    durations = durations[np.isfinite(durations)]

    if len(durations) == 0:
        raise ValueError("Cannot infer time bins from an empty duration array.")
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2.")

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(durations, quantiles).astype(np.float32).tolist()
    edges[0] = 0.0
    edges[-1] = float("inf")

    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]

    return edges


def map_time_to_bin(duration: float, bin_edges: Sequence[float]) -> int:
    for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if start <= duration < end:
            return i
    return len(bin_edges) - 2


class ColorectalSurvivalDataset(Dataset):
    """
    Patient-level multimodal survival dataset for EviFus-CRC.

    The public release assumes that CT and pathology inputs are pre-extracted
    embeddings saved as .npy files.

    Required columns in case_df:
        - patient_name (or config.patient_id_col)
        - duration
        - event

    Common path columns in the user's CSV:
        - path_ct
        - path_bingli
    """
    def __init__(
        self,
        case_df: pd.DataFrame,
        config,
        split_name: str,
        time_bin_edges: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()

        self.case_df = case_df.reset_index(drop=True).copy()
        self.config = config
        self.split_name = split_name

        self.patient_id_col = config.patient_id_col
        self.duration_col = config.duration_col
        self.event_col = config.event_col

        self.ct_feature_dim = int(config.ct_feature_dim)
        self.pathology_feature_dim = int(config.pathology_feature_dim)
        self.missing_value = float(config.missing_value)

        self.ct_feature_root = Path(config.ct_feature_root)
        self.pathology_feature_root = Path(config.pathology_feature_root)
        self.split_csv_dir = Path(config.split_csv).expanduser().resolve().parent

        self._validate_dataframe()

        if time_bin_edges is None:
            num_bins = 4 if getattr(config, "num_time_bins", -1) in {None, -1} else int(config.num_time_bins)
            self.time_bin_edges = infer_time_bins_from_training(
                durations=self.case_df[self.duration_col].values,
                num_bins=num_bins,
            )
        else:
            self.time_bin_edges = list(time_bin_edges)

        self.case_df["time_bin"] = self.case_df[self.duration_col].apply(
            lambda x: map_time_to_bin(float(x), self.time_bin_edges)
        )

    def _validate_dataframe(self) -> None:
        required_columns = [
            self.patient_id_col,
            self.duration_col,
            self.event_col,
        ]
        missing_columns = [c for c in required_columns if c not in self.case_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in case_df: {missing_columns}")

    def _get_patient_id(self, row: pd.Series) -> str:
        return str(row[self.patient_id_col])

    def _normalize_path(self, p: Path) -> Path:
        if p.is_absolute():
            return p
        return (self.split_csv_dir / p).resolve()

    def _resolve_feature_path(
        self,
        row: pd.Series,
        modality: str,
        patient_id: str,
    ) -> Optional[Path]:
        """
        Resolve the path to a modality feature file.

        Priority:
        1) If an explicit path column exists in the CSV, use it directly.
           - CT: path_ct / ct_feature_path
           - Pathology: path_bingli / pathology_feature_path
        2) Only if such columns do NOT exist in the CSV, fall back to
           root_dir / {patient_id}.npy
        """
        if modality == "ct":
            explicit_cols = ["path_ct", "ct_feature_path"]
            root_dir = self.ct_feature_root
        elif modality == "pathology":
            explicit_cols = ["path_bingli", "pathology_feature_path"]
            root_dir = self.pathology_feature_root
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # If the CSV contains explicit path columns, trust the CSV only.
        existing_explicit_cols = [col for col in explicit_cols if col in row.index]
        if existing_explicit_cols:
            for col in existing_explicit_cols:
                if pd.notna(row[col]) and str(row[col]).strip():
                    return self._normalize_path(Path(str(row[col])).expanduser())
        # Column exists but value is empty -> treat as missing modality
            return None

        # Fallback only when no explicit path columns exist in the CSV
        fallback = root_dir / f"{patient_id}.npy"
        return fallback

    def _load_ct_feature(self, path: Optional[Path]) -> Tuple[np.ndarray, int]:
        if path is None or not path.exists():
            return np.full(self.ct_feature_dim, self.missing_value, dtype=np.float32), 0

        try:
            feature = np.load(path)
            feature = reduce_to_1d_feature(feature)

            if feature.shape[0] != self.ct_feature_dim:
                raise ValueError(
                    f"CT feature dim mismatch: expected {self.ct_feature_dim}, got {feature.shape[0]}"
                )
            return feature.astype(np.float32), 1
        except Exception as exc:
            print(f"[Warning] Failed to load CT feature from {path}: {exc}")
            return np.full(self.ct_feature_dim, self.missing_value, dtype=np.float32), 0

    def _load_pathology_feature(self, path: Optional[Path]) -> Tuple[np.ndarray, int]:
        if path is None or not path.exists():
            return np.full(self.pathology_feature_dim, self.missing_value, dtype=np.float32), 0

        try:
            feature = np.load(path)
            feature = reduce_to_1d_feature(feature)

            if feature.shape[0] != self.pathology_feature_dim:
                raise ValueError(
                    f"Pathology feature dim mismatch: expected {self.pathology_feature_dim}, got {feature.shape[0]}"
                )

            feature = zscore_1d(feature)
            return feature.astype(np.float32), 1
        except Exception as exc:
            print(f"[Warning] Failed to load pathology feature from {path}: {exc}")
            return np.full(self.pathology_feature_dim, self.missing_value, dtype=np.float32), 0

    def get_time_bin_edges(self) -> List[float]:
        return list(self.time_bin_edges)

    def __len__(self) -> int:
        return len(self.case_df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.case_df.iloc[index]
        patient_id = self._get_patient_id(row)

        ct_path = self._resolve_feature_path(row=row, modality="ct", patient_id=patient_id)
        pathology_path = self._resolve_feature_path(row=row, modality="pathology", patient_id=patient_id)

        ct_feature, ct_present = self._load_ct_feature(ct_path)
        pathology_feature, pathology_present = self._load_pathology_feature(pathology_path)

        label = int(row["label"]) if "label" in row.index and pd.notna(row["label"]) else int(float(row[self.event_col]))
        duration = float(row[self.duration_col])
        event = float(row[self.event_col])
        time_bin = int(row["time_bin"])

        sample = {
            "patient_id": patient_id,
            # presence_mask is a key tensor used throughout the paper:
            # [1, 0] means CT present / pathology missing;
            # [0, 1] means CT missing / pathology present.

            "ct": torch.tensor(ct_feature, dtype=torch.float32),
            "pathology": torch.tensor(pathology_feature, dtype=torch.float32),
            "presence_mask": torch.tensor([ct_present, pathology_present], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "duration": torch.tensor(duration, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32),
            "time_bin": torch.tensor(time_bin, dtype=torch.long),
        }
        return sample


def build_datasets(config):
    df = pd.read_csv(config.split_csv)

    if config.split_col not in df.columns:
        raise ValueError(f"Column '{config.split_col}' not found in split CSV.")

    train_df = df[df[config.split_col] == config.train_tag].copy()
    val_df = df[df[config.split_col] == config.val_tag].copy()
    test_df = df[df[config.split_col] == config.test_tag].copy()

    if len(train_df) == 0:
        raise ValueError("Training split is empty.")
    if len(val_df) == 0:
        raise ValueError("Validation split is empty.")
    if len(test_df) == 0:
        raise ValueError("Test split is empty.")

    # Safety check for public release:
    # train / val / test must be patient-disjoint.
    train_ids = set(train_df[config.patient_id_col].astype(str))
    val_ids = set(val_df[config.patient_id_col].astype(str))
    test_ids = set(test_df[config.patient_id_col].astype(str))

    if train_ids & val_ids:
        raise ValueError(
            f"Train/val split leakage detected: {len(train_ids & val_ids)} overlapping patients."
        )
    if train_ids & test_ids:
        raise ValueError(
            f"Train/test split leakage detected: {len(train_ids & test_ids)} overlapping patients."
        )
    if val_ids & test_ids:
        raise ValueError(
            f"Val/test split leakage detected: {len(val_ids & test_ids)} overlapping patients."
        )

    train_set = ColorectalSurvivalDataset(
        case_df=train_df,
        config=config,
        split_name="train",
        time_bin_edges=None,
    )
    shared_edges = train_set.get_time_bin_edges()

    val_set = ColorectalSurvivalDataset(
        case_df=val_df,
        config=config,
        split_name="val",
        time_bin_edges=shared_edges,
    )
    test_set = ColorectalSurvivalDataset(
        case_df=test_df,
        config=config,
        split_name="test",
        time_bin_edges=shared_edges,
    )

    return train_set, val_set, test_set
