import pandas as pd
from sklearn.impute import KNNImputer
from tqdm import tqdm
from Libs.utils.log_utils import get_logger
from Libs.utils.exp_utils import seed_everything
from pathlib import Path  # Added to handle file system path operations
from typing import Optional

# NOTE: Prefer IterativeImputer (MICE) with optional fancyimpute acceleration.
# Fallback to sklearn implementation if fancyimpute not installed.

import pandas as pd

try:
    from fancyimpute import IterativeImputer as _IterativeImputer  # type: ignore
    _IMPUTER_BACKEND = "fancyimpute"
except ImportError:  # pragma: no cover – ensure code still works without extra dependency
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer as _IterativeImputer  # type: ignore
    _IMPUTER_BACKEND = "sklearn"

from sklearn.impute import SimpleImputer

# Initialise module-level logger
logger = get_logger(__name__)

# -----------------------------------------------------------------------------
#  Explicit dtypes to avoid pandas default float64 up‐casting of ints / bools.
#  Only commonly used columns are listed; unknown columns fall back to default.
# -----------------------------------------------------------------------------
CSV_DTYPE_MAP = {
    # Primary keys
    "subject_id": "int32",
    "hadm_id": "int32",  # Hospital admission ID
    "stay_id": "int32",  # ICU stay ID
    "hours_from_onset": "int32",
    # Demographics
    "age": "float32",
    "gender": "category",
    # Actions / outcomes if present in state CSV for future steps
    "sofa_score": "float32",
}

# Date parsing columns (for completeness; may not exist)
DATE_COLS: list[str] = []


def load_data(vitals_path: str, labres_path: str, demo_path: str, sofa_path: str):
    """Load four CSV files as DataFrames.

    Args:
        vitals_path (str): Path to the vitals CSV file.
        labres_path (str): Path to the laboratory results CSV file.
        demo_path (str): Path to the demographics CSV file.
        sofa_path (str): Path to the SOFA score CSV file.

    Returns:
        tuple: (vitals_df, labres_df, demo_df, sofa_df)
    """
    read_csv_kwargs = dict(dtype=CSV_DTYPE_MAP, parse_dates=DATE_COLS, low_memory=False)

    vitals = pd.read_csv(vitals_path, **read_csv_kwargs)
    labres = pd.read_csv(labres_path, **read_csv_kwargs)
    demo = pd.read_csv(demo_path, **read_csv_kwargs)
    sofa = pd.read_csv(sofa_path, **read_csv_kwargs)
    return vitals, labres, demo, sofa


def merge_data(vitals: pd.DataFrame, labres: pd.DataFrame, demo: pd.DataFrame, sofa: pd.DataFrame) -> pd.DataFrame:
    """Merge four DataFrames on appropriate keys.

    Args:
        vitals (pd.DataFrame): Vitals DataFrame.
        labres (pd.DataFrame): Laboratory results DataFrame.
        demo (pd.DataFrame): Demographics DataFrame.
        sofa (pd.DataFrame): SOFA score DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    merged = pd.merge(
        vitals,
        labres,
        on=["subject_id", "hours_from_onset"],
        how="left",
        suffixes=("", "_lab")
    )
    merged = pd.merge(
        merged,
        demo,
        on="subject_id",
        how="left"
    )
    merged = pd.merge(
        merged,
        sofa,
        on=["subject_id", "hours_from_onset"],
        how="left"
    )
    return merged


def select_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select core state space features, removing unrelated columns.

    Args:
        df (pd.DataFrame): Merged DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing only state-related features.
    """
    state_cols = [
        # Primary keys
        "subject_id", "hours_from_onset",
        # Vitals
        "gcs", "heart_rate", "sirs_score", "sbp", "dbp", "mbp", "temperature", "spo2",
        # Laboratory results
        "urineoutput", "creatinine", "potassium", "calcium", "chloride", "bun", "sodium", "glucose",
        "albumin", "globulin", "lactate", "pt", "ptt", "inr", "base_excess", "bicarb",
        "hemoglobin", "platelets", "wbc",
        # Demographics
        "age", "gender", "height", "weight", "bmi", "charlson_comorbidity_index", "icu_readmit",
        # SOFA score
        "sofa_score"
    ]
    # Only keep columns present in df
    state_cols = [col for col in state_cols if col in df.columns]
    return df[state_cols]


def save_df(df: pd.DataFrame, path: str):
    """Save a DataFrame to a CSV file, creating directories if necessary.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): Output file path. Both relative and absolute paths are supported.

    Raises:
        OSError: If the file cannot be written due to permission issues or other OS-level errors.
    """
    # Ensure the parent directory exists before writing the file
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def get_missing_stats(df: pd.DataFrame) -> pd.Series:
    """Calculate the missing value ratio for each column.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: Missing ratio for each column.
    """
    return df.isnull().mean()


def iterative_impute(
    df: pd.DataFrame, columns: list, max_iter: int = 10, sample_posterior: bool = False, seed: Optional[int] = None
) -> pd.DataFrame:
    """Impute specified *numeric* columns using MICE/IterativeImputer.

    Compared with :class:`sklearn.impute.KNNImputer`, MICE scales roughly
    O(n_samples × n_features) per iteration but **does not require** the full
    pairwise distance matrix and therefore avoids the O(n²) memory blow-up.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    columns : list[str]
        Target columns to impute (must be numeric).
    max_iter : int, default 10
        Number of MICE iterations.
    sample_posterior : bool, default False
        Add stochasticity for multiple imputation.
    seed : int | None
        Random seed.
    """

    if not columns:
        return df

    imputer = _IterativeImputer(
        max_iter=max_iter,
        sample_posterior=sample_posterior,
        random_state=seed,
    )

    df[columns] = imputer.fit_transform(df[columns])
    logger.info("Iterative imputation (%s) completed on %d columns.", _IMPUTER_BACKEND, len(columns))
    return df


def sample_and_hold_impute(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Impute specified columns using sample-and-hold (forward fill + mean fill for first value).

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to impute.

    Returns:
        pd.DataFrame: DataFrame after sample-and-hold imputation.
    """
    for col in columns:
        # 若首位为缺失值，使用列统计量填充：数值→均值；分类→众数/空字符串
        if pd.isnull(df[col].iloc[0]):
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].mean(skipna=True)
            else:
                mode = df[col].mode(dropna=True)
                fill_val = mode.iloc[0] if not mode.empty else ""
            df.loc[df.index[0], col] = fill_val
        # 前向填充保持时间一致性
        df[col] = df[col].ffill()
    return df


def iterative_impute_blockwise(
    df: pd.DataFrame,
    columns: list,
    max_iter: int = 10,
    block_size: int = 100_000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Block-wise Iterative Imputer to control memory on very large tables.

    This avoids loading the full *n_samples×n_features* matrix into memory at
    once by processing chunks of rows sequentially.  Because MICE estimates
    feature relations, we fit on a *random subset* of rows ("prototype") to
    learn transformers, then transform each block.
    """

    if not columns:
        return df

    # ------------------------------------------------------------------
    # Step 1) Fit on a prototype subset (<=1e5 rows or 5% of data).
    # ------------------------------------------------------------------

    n_rows = df.shape[0]
    sample_size = min(int(0.05 * n_rows), 100_000)
    prototype = df[columns].sample(n=sample_size, random_state=seed) if n_rows > sample_size else df[columns]

    base_imputer = _IterativeImputer(max_iter=max_iter, random_state=seed)
    base_imputer.fit(prototype)

    # ------------------------------------------------------------------
    # Step 2) Transform by blocks
    # ------------------------------------------------------------------

    for start in range(0, n_rows, block_size):
        end = min(start + block_size, n_rows)
        block = df.iloc[start:end][columns]
        df.loc[df.index[start:end], columns] = base_imputer.transform(block)
        logger.debug("Iterative imputer transformed rows %d ~ %d", start, end)

    return df


def impute_missing(
    df: pd.DataFrame,
    id_cols: list,
    knn_thresh: float = 0.3,
    hold_thresh: float = 0.95,
    block_size: int = 10000,
) -> pd.DataFrame:
    """Impute missing values with a *column-then-row* strategy that avoids breaking
    per-patient temporal consistency.

    Strategy
    --------
    1.  *Column-wise*:  
        • **> hold_thresh** → drop column.  
        • **≤ knn_thresh** → block-wise KNN 插值.  
        • *(knn_thresh, hold_thresh]* → sample-and-hold.
    2.  *Row-wise*:  After the above,对每位病人按照时间序列执行
        forward/backward fill (只利用历史或同列信息)。
    3.  最后若仍有缺失，只删除对应 **列** 而**不**删除整行，防止轨迹断裂。

    Args:
        df (pd.DataFrame): Input DataFrame.
        id_cols (list): Primary key columns, not imputed.
        knn_thresh (float): KNN imputation threshold.
        hold_thresh (float): Sample-and-hold threshold.
        block_size (int): Block size for KNN imputation.

    Returns:
        pd.DataFrame: Imputed DataFrame with no missing values.
    """
    # Ensure only existing ID columns are used
    id_cols = [c for c in id_cols if c in df.columns]
    
    # CRITICAL: Sort by patient and time to ensure temporal consistency
    df = df.sort_values(["subject_id", "hours_from_onset"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1) Column-wise strategy (retain original logic) --------------------
    # ------------------------------------------------------------------
    miss_rate = get_missing_stats(df)
    drop_cols = miss_rate[miss_rate > hold_thresh].index.tolist()
    impute_cols = miss_rate[(miss_rate <= knn_thresh) & (~miss_rate.index.isin(id_cols))].index.tolist()
    # 仅对数值型列使用数值插值
    impute_cols = [c for c in impute_cols if pd.api.types.is_numeric_dtype(df[c])]
    hold_cols = miss_rate[(miss_rate > knn_thresh) & (miss_rate <= hold_thresh) & (~miss_rate.index.isin(id_cols))].index.tolist()
    
    logger.info(f"Imputation strategy: drop {len(drop_cols)} cols, iterative {len(impute_cols)} cols, hold {len(hold_cols)} cols")

    # Drop columns with extremely high missing rate
    df = df.drop(columns=drop_cols)

    # Iterative imputation (blockwise) for *low* missing-rate numeric columns
    if impute_cols:
        logger.info("Iterative imputation in progress … (cols: %d)", len(impute_cols))
        df = iterative_impute_blockwise(df, impute_cols, block_size=block_size, seed=42)
        logger.info("Iterative imputation completed.")

    # Sample-and-hold for moderate missing-rate columns
    if hold_cols:
        logger.info("Sample-and-hold imputation in progress … (cols: %d)", len(hold_cols))

        # CRITICAL: Compute global statistics BEFORE patient-wise processing
        # to avoid future information leakage within patient trajectories
        global_stats = {}
        for col in hold_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                global_stats[col] = df[col].mean(skipna=True)
            else:
                mode_val = df[col].mode(dropna=True)
                global_stats[col] = mode_val.iloc[0] if not mode_val.empty else ""
        
        # Fill first observation per patient with global statistic, then ffill
        for col in hold_cols:
            fill_val = global_stats[col]
            
            # Only fill the first occurrence per patient if it's missing
            first_mask = df.groupby("subject_id")[col].transform("first").isna()
            df.loc[first_mask, col] = fill_val

        # Forward fill per patient without Python loops (temporal consistency preserved)
        df[hold_cols] = df.groupby("subject_id", group_keys=False)[hold_cols].ffill()

        logger.info("Sample-and-hold imputation completed.")

    # ------------------------------------------------------------------
    # 2) Patient-level forward fill to patch sporadic NaNs that survive step 1.
    #    We **strictly avoid backward fill** to prevent future information leakage.
    # ------------------------------------------------------------------

    df = (
        df.groupby("subject_id", as_index=False)
        .apply(lambda g: g.fillna(method="ffill"))
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 3) Final cleanup: Remove columns (except id_cols) that are still missing
    # ------------------------------------------------------------------
    remaining_na_cols = [c for c in df.columns if df[c].isna().any() and c not in id_cols]
    if remaining_na_cols:
        logger.warning("Filling residual NaNs in %d columns with fallback statistics", len(remaining_na_cols))
        for col in remaining_na_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                fill_val = mode.iloc[0] if not mode.empty else ""
                df[col] = df[col].fillna(fill_val)

    # Assert no NaN left (excluding id_cols which should never be NaN)
    assert not df.drop(columns=id_cols).isna().any().any(), "NaNs remain after imputation pipeline."

    return df


def main():
    """Main function: merge, save raw, impute, save imputed result, and keep only state-related features."""
    seed_everything(42)  # Reproducibility across pandas/sklearn

    vitals_path = "Input/raw/shared/aki_vitalsigns_timeseries.csv"
    labres_path = "Input/raw/shared/aki_labres_timeseries.csv"
    demo_path = "Input/raw/shared/aki_demographics.csv"
    sofa_path = "Input/raw/shared/sofa_aki.csv"
    out_final = "Input/processed/state.csv"

    # Load all data
    vitals, labres, demo, sofa = load_data(vitals_path, labres_path, demo_path, sofa_path)
    merged = merge_data(vitals, labres, demo, sofa)

    id_cols = ["subject_id", "hours_from_onset"]
    merged_imputed = impute_missing(merged, id_cols)
    merged_imputed = select_state_columns(merged_imputed)
    save_df(merged_imputed, out_final)
    print(f"Data saved to {out_final}")


if __name__ == "__main__":
    main() 