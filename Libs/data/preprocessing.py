"""
Unified preprocessing utilities for PoG-BVE framework.

This module consolidates the functionality previously spread across
`preprocess.py` and `filter_cohort.py` into a single location.  It exposes
 two primary classes:

1. `Preprocessor` – handles cleaning, imputation, scaling and tensor
   construction for patient trajectory data.
2. `CohortDataFilter` – ensures trajectory, state and comorbidity tables
   share the same patient cohort and offers helpers for filtering and
   validation.

Example
-------
>>> from Libs.data.preprocessing import Preprocessor, CohortDataFilter
>>> pre = Preprocessor("Input/processed/ICU_AKI")
>>> tensors = pre.run("Input/raw/trajectory_vent.csv")

>>> filt = CohortDataFilter()
>>> filt.filter_to_trajectory_cohort()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
import numpy as np
import torch

from Libs.utils.log_utils import get_logger

__all__ = ["Preprocessor", "CohortDataFilter"]


class DataPipelineError(RuntimeError):
    """Raised when a stage in the data construction pipeline fails.

    *Any* failure that makes the produced artefact unreliable (e.g. empty
    DataFrame, mismatched cohort alignment, corrupted CSV) **must** raise
    this exception so that the caller (typically
    ``Libs.scripts.build_dataset`` or the *Trainer* during on-the-fly data
    building) can abort early instead of continuing with inconsistent
    inputs.  Downstream code should *never* catch ``Exception`` broadly but
    specifically handle :class:`DataPipelineError` to ensure unrelated
    errors still propagate.
    """

    pass


class Preprocessor:
    """Basic CSV-to-tensor preprocessor for trajectory data.

    Attributes
    ----------
    od : pathlib.Path
        Output directory where processed tensors and scaler stats are saved.
    rh : int
        Resample window in hours when flooring the `charttime` stamp.
    concat : bool
        Whether to concatenate the missing-value mask to the feature tensor.
    mean/std : Dict[str, float]
        Per-feature statistics computed during scaling.
    """

    def __init__(
        self,
        out_dir: str | Path,
        resample_hours: int = 4,
        concat_mask: bool = True,
        reward_params: dict | None = None,
        *,
        backfill: bool = False,
        numeric_agg: str = "last",
    ) -> None:
        """Create a Preprocessor.

        Args:
            out_dir: Directory to save processed outputs.
            resample_hours: Temporal resolution in hours.
            concat_mask: Whether to concatenate missing-value mask to feature tensor.
            reward_params: Hyper-parameters for reward computation.
            backfill: Whether to apply backward fill after forward fill during imputation.
                默认 *False* 以避免未来信息泄漏。
            numeric_agg: Aggregation method for numeric features when多条记录落在同一
                resample 时间窗。可选 ``{"mean","last","first"}``；默认 "last" 保留最新
                观测，减少信息平滑带来的偏差。
        """

        self.od = Path(out_dir)
        self.od.mkdir(parents=True, exist_ok=True)
        self.rh = resample_hours
        self.concat = concat_mask
        self.mean: Dict[str, float] = {}
        self.std: Dict[str, float] = {}
        self.reward_params = reward_params or {}

        # --- new options ---
        self.backfill = bool(backfill)
        self.numeric_agg = numeric_agg if numeric_agg in {"mean", "last", "first"} else "last"

    # ------------------------------------------------------------------
    def run(self, csv: str | Path, save: bool = True) -> Dict[str, Any]:
        """Execute the full preprocessing pipeline.

        Parameters
        ----------
        csv : str or Path
            Path to the raw trajectory CSV file.
        save : bool, default True
            If *True* the resulting tensors and scaler statistics are
            persisted under :pyattr:`od`.
        """
        df = self._load(csv)
        miss = df.isna().astype(np.uint8)
        df = self._impute(df)
        df = self._scale(df)
        tensors = self._tensor(df, miss)

        if save:
            torch.save(tensors, self.od / "preproc_tensors.pt")
            torch.save({"mean": self.mean, "std": self.std}, self.od / "scaler_stats.pt")
        return tensors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self, fp: str | Path) -> pd.DataFrame:
        """Load raw trajectory CSV handling both timestamp and hour-indexed variants.

        This helper now supports two possible temporal columns:

        1. ``charttime`` – original datetime stamp in the raw MIMIC tables.
        2. ``hours_from_onset`` – integer hours since sepsis onset produced by
           :pyclass:`Libs.data.build_trajectory.ChunkedTrajectoryBuilder`.

        The method normalises either representation into a unified
        ``charttime`` column so that downstream preprocessing remains
        unchanged.
        """

        # df = pd.read_csv(fp)  # removed redundant full read
        header_cols = pd.read_csv(fp, nrows=0).columns
        if "charttime" in header_cols:
            df = pd.read_csv(fp, parse_dates=["charttime"], infer_datetime_format=True)
        else:
            df = pd.read_csv(fp)

        if "charttime" in df.columns:
            # Original datetime representation ➜ floor to resample window
            df["charttime"] = pd.to_datetime(df["charttime"]).dt.floor(f"{self.rh}h")

        elif "hours_from_onset" in df.columns:
            # Integer hour representation ➜ bucket into resample window and
            # convert to a *relative* timestamp for uniform downstream logic.
            # We convert to Timedelta to preserve ordering semantics while
            # avoiding bogus calendar dates.
            df["charttime"] = (
                (df["hours_from_onset"] // self.rh) * self.rh  # floor to multiple
            ).astype(int)

            # Optionally, convert to pandas Timedelta for future datetime ops
            df["charttime"] = pd.to_timedelta(df["charttime"], unit="h")

        else:
            raise ValueError(
                "CSV must contain either 'charttime' (datetime) or 'hours_from_onset' (int hours) column."
            )

        # Aggregate rows at identical (subject_id, charttime) without losing
        # categorical columns: numeric → mean, others → first.
        agg_map: dict[str, str] = {}
        for col, dtype in df.dtypes.items():
            if col in ("subject_id", "charttime"):
                continue
            if pd.api.types.is_numeric_dtype(dtype):
                agg_map[col] = self.numeric_agg
            else:
                agg_map[col] = "first"
        return (
            df.groupby(["subject_id", "charttime"], as_index=False)
            .agg(agg_map)
        )

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        # 仅 forward-fill；可选 backfill 由实例配置控制以避免信息泄漏
        if self.backfill:
            df = (
                df.groupby("subject_id", as_index=False)
                .apply(lambda g: g.fillna(method="ffill").fillna(method="bfill"))
                .reset_index(drop=True)
            )
        else:
            df = (
                df.groupby("subject_id", as_index=False)
                .apply(lambda g: g.fillna(method="ffill"))
                .reset_index(drop=True)
            )

        num_cols = [
            c
            for c in df.columns
            if c not in ("subject_id", "charttime")
            and df[c].dtype != object
            and df[c].nunique(dropna=True) > 10
        ]
        cat_cols = [c for c in df.columns if c not in ("subject_id", "charttime") and c not in num_cols]

        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        for c in cat_cols:
            mode = df[c].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else 0
            df[c] = df[c].fillna(fill)
        return df

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = [
            c
            for c in df.columns
            if c not in ("subject_id", "charttime") and df[c].dtype != object and df[c].nunique(dropna=True) > 10
        ]
        self.fit_scaler(df, feat)
        return self.transform_scaler(df, feat)

    def fit_scaler(self, df: pd.DataFrame, feat_cols: list[str]) -> None:
        """Compute并缓存均值/方差，仅应在 *训练集* 上调用。"""
        self.mean = df[feat_cols].mean().to_dict()
        self.std = df[feat_cols].std(ddof=0).replace(0, 1).to_dict()

    def transform_scaler(self, df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
        """Apply 缩放，返回新 DataFrame（不修改原 df 以防止副作用）。"""
        _df = df.copy()
        _df[feat_cols] = (_df[feat_cols] - pd.Series(self.mean)) / pd.Series(self.std)
        return _df

    def inverse_transform_scaler(self, arr: np.ndarray | torch.Tensor, feat_cols: list[str]):  # noqa: D401
        """将标准化后的数组还原到原始数值区间（便于可解释性）。"""
        if isinstance(arr, torch.Tensor):
            for i, c in enumerate(feat_cols):
                arr[..., i] = arr[..., i] * self.std.get(c, 1) + self.mean.get(c, 0)
            return arr
        else:  # assume np.ndarray
            for i, c in enumerate(feat_cols):
                arr[..., i] = arr[..., i] * self.std.get(c, 1) + self.mean.get(c, 0)
            return arr

    def _tensor(self, df: pd.DataFrame, miss: pd.DataFrame) -> Dict[str, Any]:
        feat = [c for c in df.columns if c not in ("subject_id", "charttime")]
        pids = df.subject_id.unique()
        seqs: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        lengths: List[int] = []

        from Libs.utils.data_utils import compute_sequence_rewards  # local import to avoid heavy dep when unused

        rewards: List[torch.Tensor] = []

        for pid in pids:
            g = df[df.subject_id == pid]
            seqs.append(torch.tensor(g[feat].values, dtype=torch.float32))
            masks.append(torch.tensor(miss.loc[g.index, feat].values, dtype=torch.float32))
            lengths.append(len(g))

            # Compute reward if sofa & lactate available
            if {"sofa", "lactate", "outcome_alive"}.issubset(g.columns):
                sofa = torch.tensor(g["sofa"].values)
                lact = torch.tensor(g["lactate"].values)
                outcome = torch.tensor(g["outcome_alive"].iloc[0] == 1)

                r = compute_sequence_rewards(
                    sofa.unsqueeze(0), lact.unsqueeze(0), outcome.unsqueeze(0),
                    c0=self.reward_params.get("c0", -0.025),
                    c1=self.reward_params.get("c1", -0.125),
                    c2=self.reward_params.get("c2", -2.0),
                    r_alive=self.reward_params.get("r_alive", 1.0),
                    r_dead=self.reward_params.get("r_dead", -1.0),
                )[0]
                rewards.append(r)
            else:
                rewards.append(torch.zeros(len(g)))

        T = max(lengths)

        def pad(t: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(t, (0, 0, 0, T - t.shape[0]))

        x = torch.stack([pad(s) for s in seqs])
        if self.concat:
            x = torch.cat([x, torch.stack([pad(m) for m in masks])], dim=-1)

        reward_tensor = torch.stack([pad(r.unsqueeze(1))[:, 0] for r in rewards])  # (B, T)

        if self.reward_params.get("center", False):
            reward_tensor = reward_tensor - reward_tensor.mean()

        return {
            "x": x,
            "lengths": torch.tensor(lengths),
            "patient_ids": pids,
            "reward": reward_tensor,
        }


class CohortDataFilter:
    """Filter and align patient cohorts across multiple datasets."""

    def __init__(
        self,
        input_dir: str | Path = "Input/processed",
        output_dir: str | Path = "Input/processed",
        cohort_name: str = "ICU_AKI",
        task: str = "vent",
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.cohort_name = cohort_name
        self.task = task
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def get_trajectory_patients(self, trajectory_file: str = "trajectory_vent.csv") -> Set[str]:
        trajectory_path = self.input_dir / trajectory_file
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")
        print(f"[INFO] Loading trajectory data from {trajectory_path}")
        df = pd.read_csv(trajectory_path)
        if "subject_id" not in df.columns:
            raise ValueError("Trajectory file must contain 'subject_id' column")
        pids = set(df["subject_id"].astype(str).unique())
        print(f"[INFO] Found {len(pids)} unique patients in trajectory data")
        return pids

    def filter_state_data(
        self,
        patient_ids: Set[str],
        state_file: str = "state.csv",
        output_file: str = "state_filtered.csv",
    ) -> Dict[str, Any]:
        state_path = self.input_dir / state_file
        output_path = self.output_dir / output_file
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")
        print(f"[INFO] Filtering state data from {state_path}")
        df = pd.read_csv(state_path)
        orig_rows = len(df)
        orig_pats = df["subject_id"].astype(str).nunique()
        df["subject_id"] = df["subject_id"].astype(str)
        filt = df[df["subject_id"].isin(patient_ids)]
        filt.to_csv(output_path, index=False)
        stats = {
            "original_rows": orig_rows,
            "filtered_rows": len(filt),
            "original_patients": orig_pats,
            "filtered_patients": filt["subject_id"].nunique(),
            "rows_removed": orig_rows - len(filt),
            "patients_removed": orig_pats - filt["subject_id"].nunique(),
            "retention_rate": len(filt) / orig_rows if orig_rows else 0.0,
        }
        print(
            f"[INFO] State data filtered: {stats['filtered_rows']:,} rows ({stats['filtered_patients']:,} patients)"
        )
        return stats

    def filter_comorbidity_data(
        self,
        patient_ids: Set[str],
        comorbidity_file: str = "aki_comorbidity.csv",
        output_file: str = "aki_comorbidity_filtered.csv",
        input_subdir: str = "raw",
    ) -> Dict[str, Any]:
        if input_subdir == "raw":
            comorbidity_path = (
                self.input_dir.parent / "raw" / "shared" / comorbidity_file
            )
        else:
            comorbidity_path = self.input_dir / self.cohort_name / comorbidity_file
        output_path = self.output_dir / output_file
        if not comorbidity_path.exists():
            raise FileNotFoundError(f"Comorbidity file not found: {comorbidity_path}")
        print(f"[INFO] Filtering comorbidity data from {comorbidity_path}")
        df = pd.read_csv(comorbidity_path)
        orig_rows = len(df)
        orig_pats = df["subject_id"].astype(str).nunique()
        df["subject_id"] = df["subject_id"].astype(str)
        filt = df[df["subject_id"].isin(patient_ids)]
        filt.to_csv(output_path, index=False)
        stats = {
            "original_rows": orig_rows,
            "filtered_rows": len(filt),
            "original_patients": orig_pats,
            "filtered_patients": filt["subject_id"].nunique(),
            "rows_removed": orig_rows - len(filt),
            "patients_removed": orig_pats - filt["subject_id"].nunique(),
            "retention_rate": len(filt) / orig_rows if orig_rows else 0.0,
        }
        print(
            f"[INFO] Comorbidity data filtered: {stats['filtered_rows']:,} rows ({stats['filtered_patients']:,} patients)"
        )
        return stats

    def filter_to_trajectory_cohort(
        self,
        trajectory_file: str = "trajectory_vent.csv",
        state_file: str = "state.csv",
        comorbidity_file: str = "aki_comorbidity.csv",
        state_output: str = "state_filtered.csv",
        comorbidity_output: str = "aki_comorbidity_filtered.csv",
        delete_original: bool = True,
        id_split_dir: str | Path | None = None,
    ) -> Dict[str, Any]:
        print("=" * 60)
        print(f"Cohort Data Filtering for {self.cohort_name}")
        print("=" * 60)
        pids = self.get_trajectory_patients(trajectory_file)
        print("\n-- Filtering State Data --")
        state_stats = self.filter_state_data(pids, state_file, state_output)
        print("\n-- Filtering Comorbidity Data --")
        com_stats = self.filter_comorbidity_data(pids, comorbidity_file, comorbidity_output)
        print("\n" + "=" * 60)
        print("Filtering Summary")
        print("=" * 60)
        print(f"Reference cohort size: {len(pids):,} patients")
        print(f"State data retention: {state_stats['retention_rate']:.1%}")
        print(f"Comorbidity data retention: {com_stats['retention_rate']:.1%}")
        print("Filtered files:")
        print(f"  - {self.output_dir / state_output}")
        print(f"  - {self.output_dir / comorbidity_output}")

        # ---------------- Optional post-processing ----------------
        # Delete bulky *raw* CSVs once filtered versions are produced.
        if delete_original:
            for raw_fp in [self.input_dir / state_file, self.input_dir / comorbidity_file]:
                try:
                    Path(raw_fp).unlink(missing_ok=True)
                    print(f"[INFO] Removed raw file {raw_fp} to reduce disk usage.")
                except Exception as e:
                    print(f"[WARNING] Could not delete {raw_fp}: {e}")

        # ---------------- ID split validation ----------------
        if id_split_dir is not None:
            id_split_dir = Path(id_split_dir)
            train_ids = set((id_split_dir / "train_ids.txt").read_text().split())
            val_ids   = set((id_split_dir / "val_ids.txt").read_text().split())
            test_ids  = set((id_split_dir / "test_ids.txt").read_text().split())

            union_ids = train_ids | val_ids | test_ids
            if len(union_ids) != len(pids):
                print(
                    f"[ERROR] ID union mismatch: splits={len(union_ids)} vs cohort={len(pids)}"
                )
            overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
            if overlap:
                print(f"[ERROR] Overlapping IDs across splits detected: {len(overlap)} samples")
            else:
                print("[INFO] ID split validation ✓ – mutually exclusive and exhaustive.")
        return {
            "cohort": self.cohort_name,
            "n_patients": len(pids),
            "state_filtering": state_stats,
            "comorbidity_filtering": com_stats,
        }

    def validate_cohort_alignment(
        self,
        trajectory_file: str = "trajectory_vent.csv",
        state_file: str = "state_filtered.csv",
        comorbidity_file: str = "aki_comorbidity_filtered.csv",
    ) -> bool:
        print("\n-- Validating Cohort Alignment --")
        traj_path = self.input_dir / trajectory_file
        state_path = self.output_dir / state_file
        com_path = self.output_dir / comorbidity_file
        traj_pids = set(pd.read_csv(traj_path)["subject_id"].astype(str).unique())
        state_pids = set(pd.read_csv(state_path)["subject_id"].astype(str).unique())
        com_pids = set(pd.read_csv(com_path)["subject_id"].astype(str).unique())
        ok = traj_pids == state_pids == com_pids
        print(f"Alignment check: {'✓' if ok else '✗'}")
        if not ok:
            if traj_pids != state_pids:
                print(f"State mismatch: {len(state_pids - traj_pids)} differing patients")
            if traj_pids != com_pids:
                print(f"Comorbidity mismatch: {len(com_pids - traj_pids)} differing patients")
            raise DataPipelineError("Cohort alignment validation failed – aborting preprocessing stage.")
        return ok


def create_task_cohort_filter(task: str, **kwargs) -> CohortDataFilter:
    """Create a CohortDataFilter for a specific task.
    
    Args:
        task: Task name ('vent', 'rrt', 'iv').
        **kwargs: Additional arguments passed to CohortDataFilter.
        
    Returns:
        CohortDataFilter instance configured for the task.
    """
    # Set task-specific input/output directories
    base_processed_dir = Path("Input/processed")
    
    defaults = {
        "input_dir": base_processed_dir,
        "output_dir": base_processed_dir / task,
        "task": task,
    }
    defaults.update(kwargs)
    
    return CohortDataFilter(**defaults)


if __name__ == "__main__":
    f = CohortDataFilter()
    f.filter_to_trajectory_cohort()
    exit(0 if f.validate_cohort_alignment() else 1) 