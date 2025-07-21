import pandas as pd
import numpy as np
import os
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from Libs.utils.log_utils import get_logger
from Libs.utils.exp_utils import seed_everything
from packaging import version


logger = get_logger(__name__)

# Ensure deterministic behaviour across numpy / torch / random
seed_everything(42)


class ChunkedTrajectoryBuilder:
    """Efficiently builds RL trajectory tables in chunks, suitable for large state.csv files.

    This class processes state, action, and mortality data in chunks to construct RL trajectories and compute rewards efficiently for large-scale datasets.
    """

    def __init__(self, state_path: str, action_path: str, mortality_path: str, output_path: str, 
                 chunksize: int = 50000, config_path: str = "Libs/configs/dataset.yaml"):
        """Initializes the builder with file paths and chunk size.

        Args:
            state_path (str): Path to the state CSV file (must include 'sirs_score' column).
            action_path (str): Path to the action CSV file.
            mortality_path (str): Path to the 90-day mortality CSV file.
            output_path (str): Output path for the processed trajectory CSV.
            chunksize (int): Number of rows to process per chunk.
            config_path (str): Path to the configuration YAML file.
        """
        self.state_path = state_path
        self.action_path = action_path
        self.mortality_path = mortality_path
        self.output_path = output_path
        self.chunksize = chunksize
        
        # Load configuration from YAML file
        self.config = self._load_config(config_path)
        
        # Extract reward parameters with fallback defaults
        reward_config = self.config.get('reward_shaping', {})
        self.r_alive = reward_config.get('r_alive', 1.0)
        self.r_dead = reward_config.get('r_dead', -1.0)
        
        # Track which columns have been warned about to avoid repetitive warnings
        self._warned_missing_cols = set()
        
        logger.info(f"Loaded reward configuration: r_alive={self.r_alive}, r_dead={self.r_dead}")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file.
        
        Args:
            config_path (str): Path to the configuration file.
            
        Returns:
            dict: Configuration dictionary.
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found at {config_path}, using default values")
                return {}
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            return {}

    def load_action_mortality(self):
        """Loads action and mortality tables into memory (assumed to be small enough)."""
        try:
            # Load action file
            self.action = pd.read_csv(self.action_path)
            if self.action.empty:
                raise ValueError(f"Action file is empty: {self.action_path}")
            logger.info(f"Loaded action data: {self.action.shape[0]} rows, {self.action.shape[1]} columns")
            
            # Load mortality file with error handling
            from pathlib import Path
            mortality_file = Path(self.mortality_path)
            if not mortality_file.exists():
                raise FileNotFoundError(f"Mortality file not found: {self.mortality_path}")
            
            if mortality_file.stat().st_size == 0:
                # Handle empty mortality file by creating a dummy DataFrame
                logger.warning(f"Mortality file is empty: {self.mortality_path}")
                logger.warning("Creating dummy mortality data - all patients assumed to have survived")
                
                # Get unique subject_ids from action data
                unique_subjects = self.action['subject_id'].unique()
                self.mortality = pd.DataFrame({
                    'subject_id': unique_subjects,
                    'death_within_90d': 0  # Assume all patients survived
                })
                logger.info(f"Created dummy mortality data: {self.mortality.shape[0]} rows")
            else:
                self.mortality = pd.read_csv(self.mortality_path)
                if self.mortality.empty:
                    raise ValueError(f"Mortality file is empty: {self.mortality_path}")
                logger.info(f"Loaded mortality data: {self.mortality.shape[0]} rows, {self.mortality.shape[1]} columns")
                
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"Empty or invalid CSV file. Please check your data files:\n"
                           f"- Action file: {self.action_path}\n"
                           f"- Mortality file: {self.mortality_path}\n"
                           f"Original error: {e}")
        except Exception as e:
            raise ValueError(f"Error loading action/mortality data:\n"
                           f"- Action file: {self.action_path}\n"
                           f"- Mortality file: {self.mortality_path}\n"
                           f"Original error: {e}")

    def process_chunk(self, state_chunk: pd.DataFrame) -> pd.DataFrame:
        """Processes a single chunk of state data, merges with action/mortality, computes rewards, and returns the result.

        Args:
            state_chunk (pd.DataFrame): Chunk of the state DataFrame.

        Returns:
            pd.DataFrame: Processed chunk with rewards and done flags.
        """
        # Merge with action and mortality data
        df = pd.merge(state_chunk, self.action, on=['subject_id', 'hours_from_onset'], how='inner')
        df = pd.merge(df, self.mortality, on='subject_id', how='left')

        # Fill missing mortality info (assume survivors)
        if 'death_within_90d' not in df.columns:
            logger.warning("Column 'death_within_90d' absent in mortality table – assuming survivors (0)")
            df['death_within_90d'] = 0

        # Sort by patient and time
        df = df.sort_values(['subject_id', 'hours_from_onset']).reset_index(drop=True)

        # Add intermediate and terminal rewards
        df['intermediate_reward'] = 0.0
        df['reward'] = 0.0
        df['done'] = 0

        # **Reward shaping**: Simple intermediate rewards based on SIRS deterioration/improvement.
        # Positive reward for reductions in SIRS score (clinical improvement).
        if 'sirs_score' in df.columns:
            df['sirs_delta'] = df.groupby('subject_id')['sirs_score'].diff().fillna(0)
            df['intermediate_reward'] = -0.1 * df['sirs_delta']  # negative delta → positive reward

        # Terminal rewards: Set done=1 and apply survival-based rewards at trajectory end
        for sid, group in df.groupby('subject_id'):
            idx = group.index[-1]
            if group.loc[idx, 'death_within_90d'] == 1:
                df.at[idx, 'reward'] = self.r_dead
            else:
                df.at[idx, 'reward'] = self.r_alive
            df.at[idx, 'done'] = 1

        # Drop unnecessary columns
        out = df.drop(columns=['death_within_90d', 'intermediate_reward'])

        # Type normalisation – one pass only
        int_cols = [c for c in ['subject_id', 'hours_from_onset'] if c in out.columns]
        out[int_cols] = out[int_cols].astype('Int64')
        return out

    def _get_csv_kwargs(self):
        """Return a dictionary of keyword arguments for pd.read_csv that are compatible with the
        currently installed pandas version. Older pandas versions (<1.5) do not support the
        ``dtype_backend`` keyword and may not recognise the ``pyarrow`` engine. This helper
        constructs the kwargs dynamically to maximise compatibility while still taking advantage
        of performance improvements when available.

        Returns
        -------
        dict
            Keyword arguments that can safely be passed to :func:`pandas.read_csv`.
        """
        import inspect

        csv_kwargs = {"chunksize": self.chunksize}

        # Attempt to use the ``pyarrow`` engine when available (pandas>=1.5).
        if "engine" in inspect.signature(pd.read_csv).parameters:
            try:
                if version.parse(pd.__version__) >= version.parse("1.5"):
                    import importlib
                    if importlib.util.find_spec("pyarrow") is not None:
                        csv_kwargs["engine"] = "pyarrow"
            except Exception:
                # Fallback silently if version parsing fails or pyarrow engine unavailable
                pass

        # Add ``dtype_backend`` (pandas >=2.0) only when supported.
        if "dtype_backend" in inspect.signature(pd.read_csv).parameters:
            csv_kwargs["dtype_backend"] = "pyarrow"

        return csv_kwargs

    def run(self, subject_id_path: str = None):
        """Main process: reads state data in chunks, processes and writes results. Optionally processes only a subset of subject_ids.

        Args:
            subject_id_path (str or None): If specified, only process subject_ids in this .npy file.
        """
        self.load_action_mortality()
        subject_id_set = None
        if subject_id_path is not None:
            subject_id_set = set(np.load(subject_id_path))
            logger.info("Processing only %d subject_ids.", len(subject_id_set))
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        first_chunk = True
        chunk_idx = 0
        total_rows = 0
        # Build CSV reading options in a version-safe way.
        csv_kwargs = self._get_csv_kwargs()
        # pyarrow engine is ~2-3× faster than the default C parser for wide tables

        # ------------------------------------------------------------------
        # Limit I/O by loading **only** columns required downstream.  Although
        # loading the full table is simpler, we keep memory footprint low by
        # specifying an explicit whitelist consisting of the **state_space**
        # variables plus the primary keys.  This guarantees the resulting
        # trajectory CSV carries *all* physiologic and demographic features
        # needed during graph construction.
        # ------------------------------------------------------------------

        state_space_cols = [
            # Vitals
            "gcs", "heart_rate", "sirs_score", "sbp", "dbp", "mbp", "temperature", "spo2",
            # Labs - only include commonly available ones
            "urineoutput", "creatinine", "calcium", "chloride", "bun", "sodium", "glucose",
            # Demographics
            "age", "gender", "height", "weight", "bmi", "charlson_comorbidity_index", "icu_readmit",
            # Scores
            "sofa_score",
        ]

        # Keep order while removing accidental duplicates
        seen = set()
        state_space_cols = [c for c in state_space_cols if not (c in seen or seen.add(c))]

        needed_cols = ["subject_id", "hours_from_onset"] + state_space_cols

        # --------------------------------------------------------------
        # Robustness fix: intersect requested cols with file header to
        # avoid pandas ValueError "Usecols do not match columns" when the
        # state CSV lacks certain optional biomarkers (e.g., 'potassium').
        # --------------------------------------------------------------
        try:
            file_header = pd.read_csv(self.state_path, nrows=0).columns.tolist()
        except Exception as e:
            logger.error("Failed to read header from %s: %s", self.state_path, e)
            file_header = []

        valid_cols = [c for c in needed_cols if c in file_header]
        missing_cols = sorted(set(needed_cols) - set(valid_cols))
        if missing_cols:
            logger.warning("[ChunkedTrajectoryBuilder] Requested columns absent in state CSV – they will be skipped: %s", ", ".join(missing_cols))

        if len(valid_cols) < 2:  # need at least id & hours columns
            raise ValueError("State CSV is missing all required feature columns. Aborting trajectory build.")

        csv_kwargs["usecols"] = valid_cols

        for chunk in pd.read_csv(self.state_path, **csv_kwargs):
            if subject_id_set is not None:
                chunk = chunk[chunk['subject_id'].isin(subject_id_set)]
            if chunk.empty:
                continue
            out_chunk = self.process_chunk(chunk)
            out_chunk.to_csv(self.output_path, mode='a', index=False, header=first_chunk, float_format='%.4f')
            first_chunk = False
            chunk_idx += 1
            total_rows += len(chunk)
            logger.debug(
                "[run] Processed chunk %d | rows in chunk: %d | total: %d",
                chunk_idx,
                len(chunk),
                total_rows,
            )

    def save_terminal_rewards_only(self, output_path: str, subject_id_path: str = None):
        """Save only terminal rewards (reward=r_alive or r_dead), set other rewards to 0, and write to a separate CSV. Optionally process only a subset of subject_ids.

        Args:
            output_path (str): Output file path.
            subject_id_path (str or None): If specified, only process subject_ids in this .npy file.
        """
        subject_id_set = None
        if subject_id_path is not None:
            subject_id_set = set(np.load(subject_id_path))
            logger.info("Processing only %d subject_ids (terminal rewards)", len(subject_id_set))
        first_chunk = True
        chunk_idx = 0
        total_rows = 0
        # Ensure CSV reading options are defined before use (version-safe)
        csv_opt = self._get_csv_kwargs()
        for chunk in pd.read_csv(self.output_path, **csv_opt):
            if subject_id_set is not None:
                chunk = chunk[chunk['subject_id'].isin(subject_id_set)]
            if chunk.empty:
                continue
            chunk['reward_terminal_only'] = chunk['reward'].where(
                chunk['reward'].isin([self.r_alive, self.r_dead]), 0.0)

            # Single-pass dtype casting for key columns
            int_cols = [c for c in ['subject_id', 'hours_from_onset'] if c in chunk.columns]
            chunk[int_cols] = chunk[int_cols].astype('Int64')
            df_terminal = chunk[['subject_id', 'hours_from_onset', 'reward_terminal_only']]
            df_terminal.to_csv(output_path, mode='a', index=False, header=first_chunk, float_format='%.4f')
            first_chunk = False
            chunk_idx += 1
            total_rows += len(chunk)
            unique_subjects = df_terminal['subject_id'].nunique() if 'subject_id' in df_terminal.columns else 0
            logger.debug(
                "[save_terminal_rewards_only] %s | Chunk %d | rows: %d | written: %d | unique subj: %d | total rows: %d",
                output_path,
                chunk_idx,
                len(chunk),
                len(df_terminal),
                unique_subjects,
                total_rows,
            )

    def save_terminal_rewards_only_aligned(self, output_path: str, subject_id_list: list = None, subject_id_path: str = None):
        """Save only terminal rewards for subject_ids in subject_id_list or subject_id_path, generating an aligned terminal_rewards_vent.csv.

        Args:
            output_path (str): Output file path.
            subject_id_list (list or None): Only keep these subject_ids.
            subject_id_path (str or None): If specified, only process subject_ids in this .npy file.
        Raises:
            ValueError: If neither subject_id_list nor subject_id_path is provided.
        """
        subject_id_set = None
        if subject_id_list is not None:
            subject_id_set = set(subject_id_list)
            logger.info("Processing only %d subject_ids (aligned terminal rewards, from list)", len(subject_id_set))
        elif subject_id_path is not None:
            subject_id_set = set(np.load(subject_id_path))
            logger.info(
                "Processing only %d subject_ids (aligned terminal rewards, from npy)",
                len(subject_id_set),
            )
        else:
            raise ValueError("Must provide subject_id_list or subject_id_path")
        first_chunk = True
        chunk_idx = 0
        total_rows = 0
        # Ensure CSV reading options are defined before use (version-safe)
        csv_opt = self._get_csv_kwargs()
        for chunk in pd.read_csv(self.output_path, **csv_opt):
            chunk = chunk[chunk['subject_id'].isin(subject_id_set)]
            if chunk.empty:
                continue
            chunk['reward_terminal_only'] = chunk['reward'].where(
                chunk['reward'].isin([self.r_alive, self.r_dead]), 0.0)

            # Single-pass dtype casting for key columns
            int_cols = [c for c in ['subject_id', 'hours_from_onset'] if c in chunk.columns]
            chunk[int_cols] = chunk[int_cols].astype('Int64')
            df_terminal = chunk[['subject_id', 'hours_from_onset', 'reward_terminal_only']]
            unique_subjects = df_terminal['subject_id'].nunique() if 'subject_id' in df_terminal.columns else 0
            df_terminal.to_csv(output_path, mode='a', index=False, header=first_chunk, float_format='%.4f')
            first_chunk = False
            chunk_idx += 1
            total_rows += len(chunk)
            logger.debug(
                "[save_terminal_rewards_only_aligned] %s | Chunk %d | rows: %d | written: %d | unique subj: %d | total rows: %d",
                output_path,
                chunk_idx,
                len(chunk),
                len(df_terminal),
                unique_subjects,
                total_rows,
            )


if __name__ == "__main__":
    builder = ChunkedTrajectoryBuilder(
        state_path="Input/processed/state.csv",
        action_path="Input/raw/task/aki_vent.csv",
        mortality_path="Input/raw/shared/aki_90day_mortality.csv",
        output_path="Input/processed/trajectory_vent.csv",
        chunksize=50000
    )
    # Generate full trajectory
    builder.run(subject_id_path=None)
    # Generate full terminal_rewards
    builder.save_terminal_rewards_only(
        "Input/processed/terminal_rewards_vent.csv", subject_id_path=None) 