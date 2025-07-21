"""build_graph.py

This module provides utilities for constructing patient similarity graphs based on comorbidity data and patient trajectories.
It includes tools for loading comorbidity data, building diagnosis matrices, computing similarity, and constructing k-NN graphs.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Union
import networkx as nx
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import logging
from scipy.sparse import coo_matrix, csr_matrix
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Hyper-parameters used in similarity calculation.  Externalising into a
# dataclass makes grid-search & reproducibility easier than hard-coded kwargs.
# -----------------------------------------------------------------------------

@dataclass
class SimilarityParams:
    """Container for similarity function hyper-parameters."""

    a: float = 5.0  # Weight applied to shared diagnoses
    b: float = 5.0  # Penalty for total diagnosis burden
    c: float = 1e-3  # Rare diagnosis adjustment constant
    nonlinear_penalty: bool = False

class ComorbidityGraphBuilder:
    """Utility class for building comorbidity-based patient graphs."""

    @staticmethod
    def load_comorbidity_data(filepath: str) -> pd.DataFrame:
        """Loads comorbidity data from a CSV file (long format).

        Args:
            filepath (str): Path to the comorbidity CSV file.

        Returns:
            pd.DataFrame: DataFrame with columns ['patient_id', 'diagnosis'].
        """
        logger.info(f"Loading comorbidity data from {filepath}")
        df = pd.read_csv(filepath)
        if not {'patient_id', 'diagnosis'}.issubset(df.columns):
            raise ValueError("Input CSV must contain 'patient_id' and 'diagnosis' columns.")
        return df

    @staticmethod
    def build_diagnosis_matrix(
        df: pd.DataFrame, *, sparse: bool = True
    ) -> Tuple[Union[csr_matrix, np.ndarray], List[str], List[str]]:
        """Vectorised conversion of long-format comorbidity table → multi-hot matrix.

        This implementation avoids Python-level loops by leveraging a sparse COO
        representation.  It handles >10^5 rows in seconds versus several minutes
        for the previous :pyfunc:`iterrows` loop.

        Parameters
        ----------
        df : pd.DataFrame
            Input table with columns ``patient_id`` and ``diagnosis``.
        sparse : bool, default True
            Return a SciPy CSR matrix instead of a dense ``ndarray`` to save
            memory when the number of unique diagnoses is large.

        Returns
        -------
        diagnosis_matrix : Union[csr_matrix, np.ndarray]
            (N_patients, N_diagnoses) multi-hot matrix.
        patient_ids : list[str]
            Unique patient IDs (row order).
        diagnosis_list : list[str]
            Unique diagnosis codes (column order).
        """

        logger.info("Building diagnosis matrix (vectorised)…")

        # Preserve first-appearance order to keep deterministic mapping
        patient_ids = df["patient_id"].astype(str).drop_duplicates().tolist()
        diagnosis_list = df["diagnosis"].astype(str).drop_duplicates().tolist()

        # Map categorical codes → integer indices via pandas Categorical (fast)
        row_idx = pd.Categorical(df["patient_id"].astype(str), categories=patient_ids).codes
        col_idx = pd.Categorical(df["diagnosis"].astype(str), categories=diagnosis_list).codes

        data = np.ones(len(df), dtype=np.int8)
        coo = coo_matrix(
            (data, (row_idx, col_idx)), shape=(len(patient_ids), len(diagnosis_list)), dtype=np.int8
        )

        if sparse:
            return coo.tocsr(), patient_ids, diagnosis_list
        else:
            return coo.toarray(), patient_ids, diagnosis_list

    @staticmethod
    def compute_diagnosis_occurrence(diagnosis_matrix: np.ndarray) -> np.ndarray:
        """Counts the occurrence of each diagnosis across all patients.

        Args:
            diagnosis_matrix (np.ndarray): (N_patients, N_diagnoses) multi-hot numpy array.

        Returns:
            np.ndarray: 1D array of length N_diagnoses, each entry is the count.
        """
        logger.info("Computing diagnosis occurrence...")
        occ = diagnosis_matrix.sum(axis=0)
        # Handle sparse matrix case which returns np.matrix
        if hasattr(occ, "A1"):
            occ = occ.A1  # convert to 1-D ndarray
        return np.asarray(occ).ravel()

    @staticmethod
    def compute_similarity_matrix(
        diagnosis_matrix: Union[csr_matrix, np.ndarray],
        diagnosis_occurrence: np.ndarray,
        params: SimilarityParams = SimilarityParams(),
        **legacy_kwargs,
    ) -> np.ndarray:
        """Computes patient-patient similarity matrix based on comorbidities.

        Parameters
        ----------
        diagnosis_matrix : Union[csr_matrix, np.ndarray]
            Multi-hot encoding from :pyfunc:`build_diagnosis_matrix`.
        diagnosis_occurrence : np.ndarray
            Per-diagnosis prevalence counts.
        params : SimilarityParams, optional
            Tunable coefficients controlling the similarity function.

        Returns
        -------
        np.ndarray
            Dense (N×N) similarity matrix.
        """

        logger.info("Computing similarity matrix…")

        # ------------------------------------------------------------------
        # Backward-compat: allow callers to still pass a=…, b=…, c=… directly.
        # If any such keys appear in **legacy_kwargs, override the dataclass.
        # ------------------------------------------------------------------

        if legacy_kwargs:
            merged_kwargs = {k: getattr(params, k) for k in ("a", "b", "c", "nonlinear_penalty")}
            merged_kwargs.update({k: v for k, v in legacy_kwargs.items() if k in merged_kwargs})
            params = SimilarityParams(**merged_kwargs)

        # Validate input parameters
        if params.a < 0 or params.b < 0 or params.c < 0:
            raise ValueError("Similarity parameters a, b, c must be non-negative")

        # Ensure we can treat diagnosis_matrix with unified API
        is_sparse = isinstance(diagnosis_matrix, csr_matrix)
        
        # Validate diagnosis occurrence
        if len(diagnosis_occurrence) == 0:
            raise ValueError("diagnosis_occurrence cannot be empty")
        
        if np.any(diagnosis_occurrence < 0):
            raise ValueError("diagnosis_occurrence must be non-negative")

        # Column-wise rare disease weight with numerical stability
        # w_j = 1 / max(occ_j, min_occ) + c to prevent division by zero
        min_occurrence = 1e-8
        rare_weight = 1.0 / np.maximum(diagnosis_occurrence, min_occurrence) + params.c
        
        # Clamp weights to prevent numerical instability
        rare_weight = np.clip(rare_weight, 0, 1000.0)

        if is_sparse:
            # Scale each column by rare_weight without densifying diag matrix
            weighted = diagnosis_matrix.multiply(rare_weight)
            shared = (weighted @ diagnosis_matrix.T).toarray()
            diag_sum = np.asarray(diagnosis_matrix.sum(axis=1)).astype(float)  # shape (N,1)
        else:
            shared = diagnosis_matrix @ np.diag(rare_weight) @ diagnosis_matrix.T
            diag_sum = diagnosis_matrix.sum(axis=1, keepdims=True)

        # Handle edge case where patient has no diagnoses
        diag_sum = np.maximum(diag_sum, 0.0)

        # Penalty for diagnosis count per pair
        if params.nonlinear_penalty:
            all_diag = (diag_sum ** 2) + (diag_sum.T ** 2)
        else:
            all_diag = diag_sum + diag_sum.T

        # Compute similarity with numerical stability
        similarity = params.a * shared - params.b * all_diag
        similarity = np.maximum(similarity, 0.0)
        
        # Ensure diagonal elements are reasonable (self-similarity)
        n_patients = similarity.shape[0]
        for i in range(n_patients):
            if diag_sum[i] > 0:  # Patient has diagnoses
                # Self-similarity should be positive
                self_shared = params.a * np.sum(rare_weight * diagnosis_matrix[i].toarray().flatten()**2 if is_sparse else rare_weight * diagnosis_matrix[i]**2)
                self_penalty = params.b * (2 * diag_sum[i] if not params.nonlinear_penalty else 2 * diag_sum[i]**2)
                similarity[i, i] = max(0.0, self_shared - self_penalty)
        
        # Final numerical stability check
        similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Similarity matrix computed: shape {similarity.shape}, "
                   f"mean={similarity.mean():.4f}, max={similarity.max():.4f}")

        return similarity

    @staticmethod
    def build_knn_graph(similarity_matrix: np.ndarray, k: int = 3) -> Dict[int, List[int]]:
        """Builds k-NN graph from similarity matrix.

        Args:
            similarity_matrix (np.ndarray): (N_patients, N_patients) similarity matrix.
            k (int): Number of neighbors per node.

        Returns:
            Dict[int, List[int]]: Dict mapping node index to list of neighbor indices.
        """
        logger.info(f"Building k-NN graph with k={k}...")
        N = similarity_matrix.shape[0]
        knn_graph: Dict[int, List[int]] = {}
        for i in tqdm(range(N), desc="k-NN Graph"):
            sim_row = similarity_matrix[i].copy()
            # Exclude self
            sim_row[i] = -np.inf
            if k < N:
                nbr_idx = np.argpartition(-sim_row, k)[:k]
                # sort these k neighbours by similarity descending
                nbr_idx = nbr_idx[np.argsort(sim_row[nbr_idx])[::-1]]
            else:
                nbr_idx = np.argsort(sim_row)[::-1][:k]
            knn_graph[i] = nbr_idx.tolist()
        return knn_graph

    @staticmethod
    def knn_graph_to_networkx(knn_graph: Dict[int, List[int]], patient_ids: List[str], directed: bool = True) -> nx.Graph:
        """Converts k-NN graph dict to a NetworkX graph.

        Args:
            knn_graph (Dict[int, List[int]]): Dict mapping node index to neighbor indices.
            patient_ids (List[str]): List of patient IDs (node labels).
            directed (bool): If True, return DiGraph; else, Graph.

        Returns:
            nx.Graph: NetworkX Graph object with patient IDs as node labels.
        """
        logger.info("Converting k-NN graph to NetworkX graph...")
        G = nx.DiGraph() if directed else nx.Graph()
        for i, neighbors in knn_graph.items():
            for j in neighbors:
                G.add_edge(patient_ids[i], patient_ids[j])
        return G

    @staticmethod
    def load_comorbidity_data_wide(filepath: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """Loads comorbidity data from a wide-format CSV file.

        Args:
            filepath (str): Path to the comorbidity CSV file (wide format).

        Returns:
            Tuple[np.ndarray, List[str], List[str]]: diagnosis_matrix (N_patients, N_comorbidities),
                patient_ids (row order), diagnosis_list (column order).
        """
        logger.info(f"Loading wide-format comorbidity data from {filepath}")
        df = pd.read_csv(filepath)
        patient_ids = df['subject_id'].astype(str).tolist()
        diagnosis_list = [col for col in df.columns if col != 'subject_id']
        diagnosis_matrix = df[diagnosis_list].values.astype(int)
        return diagnosis_matrix, patient_ids, diagnosis_list

# =====================
# Node feature and label processing
# =====================

def load_patient_state_features(state_csv: str, patient_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Aggregate patient state features for each patient (e.g., t=0 or earliest).

    Args:
        state_csv (str): Path to state features CSV.
        patient_ids (List[str]): List of patient IDs to extract.

    Returns:
        Tuple[np.ndarray, List[str]]: features (N_valid, D), valid_patient_ids.
    """
    print(f"[INFO] Loading patient state features from {state_csv}")
    df = pd.read_csv(state_csv)
    df = df[df['subject_id'].astype(str).isin(patient_ids)]
    features: List[np.ndarray] = []
    valid_patient_ids: List[str] = []
    missing_patients: List[str] = []
    for pid in tqdm(patient_ids, desc="State Features"):
        sub = df[df['subject_id'].astype(str) == str(pid)]
        if sub.empty:
            missing_patients.append(pid)
            continue
        if (sub['hours_from_onset'] == 0).any():
            row = sub[sub['hours_from_onset'] == 0].iloc[0]
        else:
            row = sub.sort_values('hours_from_onset').iloc[0]
        feat = row.drop(['subject_id', 'hours_from_onset']).values.astype(np.float32)
        features.append(feat)
        valid_patient_ids.append(pid)
    if missing_patients:
        print(f"[WARNING] {len(missing_patients)} patients have no state features and will be skipped: {missing_patients[:5]}{'...' if len(missing_patients)>5 else ''}")
    return np.stack(features, axis=0), valid_patient_ids

def load_patient_labels(label_csv: str, patient_ids: List[str]) -> np.ndarray:
    """Load patient labels and align to patient_ids order.

    Args:
        label_csv (str): Path to label CSV.
        patient_ids (List[str]): List of patient IDs.

    Returns:
        np.ndarray: Labels aligned to patient_ids order.
    """
    print(f"[INFO] Loading patient labels from {label_csv}")
    df = pd.read_csv(label_csv)
    label_map = dict(zip(df['subject_id'].astype(str), df['death_within_90d']))
    labels = [label_map.get(str(pid), 0) for pid in tqdm(patient_ids, desc="Labels Align")]
    return np.array(labels, dtype=np.int64)

# =====================
# PyG edge and graph construction
# =====================

def knn_graph_to_edge_index(knn_graph: Dict[int, List[int]], undirected: bool = True) -> np.ndarray:
    """Convert k-NN graph dict to edge_index for PyG.

    Args:
        knn_graph (Dict[int, List[int]]): Dict mapping node idx to neighbor idx list.
        undirected (bool): If True, add reverse edges.

    Returns:
        np.ndarray: edge_index (2, E).
    """
    print("[INFO] Converting k-NN graph to edge_index for PyG...")
    edge_set = set()
    for src, nbrs in tqdm(knn_graph.items(), desc="Edge Index"):
        for dst in nbrs:
            edge_set.add((src, dst))
            if undirected:
                edge_set.add((dst, src))
    edge_index = np.array(list(edge_set)).T if edge_set else np.empty((2, 0), dtype=int)
    return edge_index

# =====================
# Trajectory and graph data construction
# =====================

def encode_patient_trajectories(
    df: pd.DataFrame,
    feature_cols: List[str],
    patient_ids: List[str],
    T_max: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Encode each patient's trajectory as a fixed-length tensor.

    Returns:
        x (N, T_max, D), lengths (N,), mask (N, T_max), patient_ids (N,)
    """
    if T_max <= 0:
        raise ValueError("T_max must be positive")
    
    if len(feature_cols) == 0:
        raise ValueError("feature_cols cannot be empty")
    
    if len(patient_ids) == 0:
        raise ValueError("patient_ids cannot be empty")
    
    # Validate that all feature columns exist in the dataframe
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in dataframe: {missing_cols}")
    
    features: List[np.ndarray] = []
    lengths: List[int] = []
    valid_patient_ids: List[str] = []
    D: int = len(feature_cols)
    
    for pid in patient_ids:
        sub = df[df['subject_id'] == pid].sort_values('hours_from_onset')
        
        if sub.empty:
            print(f"[WARNING] Patient {pid} has no trajectory data, skipping")
            continue
            
        # Extract trajectory features
        traj = sub[feature_cols].values
        
        # Check for invalid values
        if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
            n_invalid = np.sum(np.isnan(traj)) + np.sum(np.isinf(traj))
            print(f"[WARNING] Patient {pid} has {n_invalid} invalid values in trajectory, filling with 0")
            traj = np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)
        
        L = len(traj)
        lengths.append(L)
        
        # Pad or truncate trajectory
        if L < T_max:
            pad = np.zeros((T_max - L, D))
            traj = np.vstack([traj, pad])
        else:
            traj = traj[:T_max]
            
        features.append(traj)
        valid_patient_ids.append(pid)
    
    if len(features) == 0:
        raise ValueError("No valid patient trajectories found")
    
    x = np.stack(features)
    lengths_arr = np.array(lengths)
    mask = np.zeros((len(valid_patient_ids), T_max), dtype=np.float32)
    
    for i, l in enumerate(lengths_arr):
        mask[i, :min(l, T_max)] = 1.0
    
    # Final validation
    if np.any(lengths_arr == 0):
        n_empty = int(np.sum(lengths_arr == 0))
        print(f"[WARNING] {n_empty} patients had empty trajectories after processing")
        
        # Remove patients with empty trajectories
        valid_mask = lengths_arr > 0
        if np.any(valid_mask):
            x = x[valid_mask]
            lengths_arr = lengths_arr[valid_mask]
            mask = mask[valid_mask]
            valid_patient_ids = [pid for pid, valid in zip(valid_patient_ids, valid_mask) if valid]
        else:
            raise ValueError("All patients have empty trajectories")
    
    print(f"[INFO] Successfully encoded {len(valid_patient_ids)} patient trajectories")
    print(f"[INFO] Trajectory shape: {x.shape}, avg length: {lengths_arr.mean():.1f}")
    
    return x, lengths_arr, mask, valid_patient_ids

def build_knn_graph_from_comorbidity(
    comorbidity_csv: str,
    k: int
) -> Tuple[np.ndarray, List[str]]:
    """Build k-NN graph edge_index from comorbidity wide table.

    Args:
        comorbidity_csv (str): Path to comorbidity CSV.
        k (int): Number of neighbors.

    Returns:
        Tuple[np.ndarray, List[str]]: edge_index (2, E), patient_ids
    """
    comorb = pd.read_csv(comorbidity_csv)
    diag_cols = [c for c in comorb.columns if c != 'subject_id']
    diag_matrix_dense = comorb[diag_cols].values.astype(int)
    # Convert to sparse CSR to save memory (many zeros expected)
    diag_matrix = csr_matrix(diag_matrix_dense)
    patient_ids = comorb['subject_id'].astype(str).tolist()

    # ------------------------------------------------------------------
    # Use nearest-neighbour search directly to avoid materialising full
    # similarity matrix.  ``metric='cosine'`` returns *distance* in [0,2],
    # therefore we take the smallest distances (most similar).
    # ------------------------------------------------------------------

    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(diag_matrix)
    distances, indices = nn.kneighbors(diag_matrix, return_distance=True)

    # Build undirected edge list excluding self-loops (first neighbour)
    edge_list: List[Tuple[int, int]] = []
    for src, neigh in enumerate(indices):
        for dst in neigh[1:]:  # skip self index at position 0
            edge_list.append((src, dst))
            edge_list.append((dst, src))

    edge_index = np.array(edge_list, dtype=int).T if edge_list else np.empty((2, 0), dtype=int)
    return edge_index, patient_ids

def build_trajectory_graph(
    traj_csv: str,
    comorbidity_csv: str,
    action_cols: List[str] = None,  # Will be determined from TaskManager or dataset config
    k: int = 3,
    T_max: int = 30,
    save_path: str = "Input/processed/patient_traj_graph.pt",
    mortality_csv: str = "Input/raw/shared/aki_90day_mortality.csv"
) -> Data:
    """Build a patient similarity graph with trajectory features as node attributes.

    Returns:
        Data: PyTorch Geometric Data object with all fields strictly synchronized.
    """
    # Determine action columns if not provided
    if action_cols is None:
        try:
            from Libs.utils.task_manager import get_current_task_config
            task_config = get_current_task_config()
            action_cols = task_config.action_cols
            print(f"[INFO] Using action columns from TaskManager: {action_cols}")
        except:
            # Fallback to vent task columns for backward compatibility
            action_cols = ['peep_bin', 'fio2_bin', 'tidal_volume_ibw_bin']
            print(f"[INFO] Using fallback action columns: {action_cols}")
    
    print("[STEP 1] Reading comorbidity table...")
    comorb = pd.read_csv(comorbidity_csv)

    print("[STEP 2] Reading trajectory table and aligning order...")
    df = pd.read_csv(traj_csv)
    df['subject_id'] = df['subject_id'].astype(str)

    # ---------------- Intersection filtering with detailed logging ----------------
    # Some patients may have comorbidity data but no trajectory rows. To avoid
    # subsequent length==0 removal, restrict to the intersection early.
    traj_ids = set(df['subject_id'])
    comorb_ids = set(comorb['subject_id'].astype(str))
    intersect_ids = traj_ids & comorb_ids
    
    # Log alignment statistics for debugging
    print(f"[INFO] Trajectory patients: {len(traj_ids)}")
    print(f"[INFO] Comorbidity patients: {len(comorb_ids)}")
    print(f"[INFO] Intersection patients: {len(intersect_ids)}")
    
    # Check if trajectory patients are missing from comorbidity data
    missing_from_comorb = traj_ids - comorb_ids
    if missing_from_comorb:
        print(f"[ERROR] {len(missing_from_comorb)} trajectory patients missing from comorbidity data")
        if len(missing_from_comorb) <= 10:
            print(f"[ERROR] Missing patient IDs: {list(missing_from_comorb)}")
        else:
            print(f"[ERROR] Missing patient IDs (first 10): {list(missing_from_comorb)[:10]}")
        # This is a critical error - trajectory patients must have comorbidity data
        raise ValueError(f"Critical: {len(missing_from_comorb)} trajectory patients missing from comorbidity data")
    
    # Check if comorbidity patients are missing from trajectory data (this is OK)
    missing_from_traj = comorb_ids - traj_ids
    if missing_from_traj:
        print(f"[INFO] {len(missing_from_traj)} comorbidity patients not in trajectory data (this is normal)")
        if len(missing_from_traj) <= 10:
            print(f"[INFO] Extra comorbidity patients: {list(missing_from_traj)}")
    
    # Calculate coverage rates
    traj_coverage = len(intersect_ids) / len(traj_ids) if traj_ids else 0
    comorb_coverage = len(intersect_ids) / len(comorb_ids) if comorb_ids else 0
    
    print(f"[INFO] Trajectory coverage: {traj_coverage:.2%} ({len(intersect_ids)}/{len(traj_ids)})")
    print(f"[INFO] Comorbidity coverage: {comorb_coverage:.2%} ({len(intersect_ids)}/{len(comorb_ids)})")
    
    # Only require that trajectory patients have comorbidity data (100% coverage)
    if traj_coverage < 1.0:
        print(f"[ERROR] Trajectory coverage too low: {traj_coverage:.2%}")
        raise ValueError("Patient ID alignment failed - trajectory patients missing from comorbidity data")
    
    print("[INFO] Patient ID alignment successful - all trajectory patients have comorbidity data")

    comorb = comorb[comorb['subject_id'].astype(str).isin(intersect_ids)].reset_index(drop=True)
    df = df[df['subject_id'].isin(intersect_ids)].reset_index(drop=True)

    comorb_patient_ids = comorb['subject_id'].astype(str).tolist()

    # Ensure df follows the same patient ordering for deterministic graph
    df['subject_id'] = pd.Categorical(df['subject_id'], categories=comorb_patient_ids, ordered=True)
    df = df.sort_values('subject_id')
    patient_ids = comorb_patient_ids  # aligned order

    # ------------------------------------------------------------------
    # Explicitly define state and action spaces
    #   • *state_cols*    – physiological and demographic variables
    #   • *action_cols*   – clinician intervention bins (passed as arg)
    #   Reward / done flags **MUST NOT** leak into the state representation.
    # ------------------------------------------------------------------

    all_cols = list(df.columns)

    # Remove duplicates while preserving original column order
    def _unique_preserve(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    # Derive candidate state columns by exclusion then enforce uniqueness
    state_cols = _unique_preserve(
        [c for c in all_cols if c not in ['subject_id', 'hours_from_onset'] + action_cols + ['reward', 'done']]
    )

    # Convert categorical columns to numeric representations when necessary
    if 'gender' in df.columns:
        gender_map = {'M': 1, 'F': 0, 'Male': 1, 'Female': 0}
        df['gender'] = df['gender'].map(gender_map).fillna(0).astype(float)

    # Final feature ordering: states followed by actions (no reward/done)
    feature_cols = state_cols + action_cols

    # Indices of each action dimension within the concatenated feature vector
    action_indices = [feature_cols.index(c) for c in action_cols]

    # === Normalize state features ===
    print("[INFO] Standardizing and normalizing state features (numeric columns only)...")
    numeric_state_cols = [c for c in state_cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_state_cols:
        state_mean = df[numeric_state_cols].mean()
        state_std = df[numeric_state_cols].std().replace(0, 1)
        df[numeric_state_cols] = (df[numeric_state_cols] - state_mean) / state_std

    print(f"[INFO] Action columns: {action_cols}, indices in feature vector: {action_indices}")

    print("[STEP 3] Encoding trajectories as fixed-length tensors...")
    x, lengths, mask, filtered_patient_ids = encode_patient_trajectories(df, feature_cols, patient_ids, T_max)

    print("[STEP 4] Building k-NN graph structure...")
    edge_index, comorb_patient_ids_check = build_knn_graph_from_comorbidity(comorbidity_csv, k)
    # Filter edge_index and comorb_patient_ids_check to keep only valid nodes
    valid_idx_set = set(filtered_patient_ids)
    idx_map = {pid: i for i, pid in enumerate(filtered_patient_ids)}
    # Find valid indices in the original comorb_patient_ids_check
    valid_indices = [comorb_patient_ids_check.index(pid) for pid in filtered_patient_ids]
    # Map from old index to new index
    old_to_new = {old: new for new, old in enumerate(valid_indices)}
    # Filter edge_index
    mask_edge = np.isin(edge_index[0], valid_indices) & np.isin(edge_index[1], valid_indices)
    edge_index_filtered = edge_index[:, mask_edge]
    # Remap to new indices
    edge_index_remap = np.vectorize(old_to_new.get)(edge_index_filtered).astype(int)
    # Check order
    assert filtered_patient_ids == [comorb_patient_ids_check[i] for i in valid_indices], (
        "Patient ID order mismatch after filtering length==0."
    )

    print("[STEP 5] Assembling PyG Data object and saving...")
    actions = x[:, :, action_indices]  # (N, T_max, n_action)
    # Load survival labels
    survival = load_patient_labels(mortality_csv, filtered_patient_ids)  # shape (N,)
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),  # (N, T_max, D)
        actions=torch.tensor(actions, dtype=torch.long),  # (N, T_max, n_action)
        mask=torch.tensor(mask, dtype=torch.float32),  # (N, T_max)
        lengths=torch.tensor(lengths, dtype=torch.long),  # (N,)
        edge_index=torch.tensor(edge_index_remap, dtype=torch.long),
        patient_ids=filtered_patient_ids,
        action_cols=action_cols,
        action_indices=action_indices,
        survival=torch.tensor(survival, dtype=torch.long)  # 0=survived, 1=died within 90 days
    )
    torch.save(data, save_path)
    print(f"[SUCCESS] PyG trajectory graph saved to {save_path}")
    return data

def validate_trajectory_graph_integrity(data: Data) -> Dict[str, Any]:
    """Comprehensive validation of trajectory graph data integrity.
    
    Args:
        data: PyTorch Geometric Data object
        
    Returns:
        Dict with validation results and statistics
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    try:
        # Basic structure validation
        required_attrs = ['x', 'actions', 'mask', 'lengths', 'edge_index', 'patient_ids', 'survival']
        for attr in required_attrs:
            if not hasattr(data, attr):
                validation_results['errors'].append(f"Missing required attribute: {attr}")
                validation_results['valid'] = False
        
        if not validation_results['valid']:
            return validation_results
            
        # Dimension consistency checks
        n_patients = len(data.patient_ids)
        T_max = data.x.shape[1]
        n_features = data.x.shape[2]
        
        validation_results['statistics'].update({
            'n_patients': n_patients,
            'T_max': T_max,
            'n_features': n_features,
            'n_edges': data.edge_index.shape[1],
            'avg_trajectory_length': float(data.lengths.float().mean()),
            'survival_rate': float(1 - data.survival.float().mean())
        })
        
        # Validate shapes
        if data.x.shape[0] != n_patients:
            validation_results['errors'].append(f"x shape mismatch: {data.x.shape[0]} vs {n_patients}")
            
        if data.mask.shape != (n_patients, T_max):
            validation_results['errors'].append(f"mask shape mismatch: {data.mask.shape} vs ({n_patients}, {T_max})")
            
        if len(data.lengths) != n_patients:
            validation_results['errors'].append(f"lengths mismatch: {len(data.lengths)} vs {n_patients}")
            
        # Validate edge indices
        if data.edge_index.shape[0] != 2:
            validation_results['errors'].append(f"edge_index should have 2 rows, got {data.edge_index.shape[0]}")
            
        if data.edge_index.max() >= n_patients:
            validation_results['errors'].append(f"edge_index contains invalid node indices")
            
        # Validate data ranges
        if torch.any(data.lengths > T_max):
            validation_results['warnings'].append("Some trajectory lengths exceed T_max")
            
        if torch.any(data.lengths == 0):
            validation_results['warnings'].append("Some patients have zero-length trajectories")
            
        # Check for NaN/Inf values
        if torch.any(torch.isnan(data.x)):
            validation_results['errors'].append("NaN values detected in features")
            
        if torch.any(torch.isinf(data.x)):
            validation_results['errors'].append("Infinite values detected in features")
            
        # Validate survival labels
        if not torch.all((data.survival == 0) | (data.survival == 1)):
            validation_results['errors'].append("Survival labels must be 0 or 1")
            
        # Check mask consistency
        for i in range(n_patients):
            length = data.lengths[i]
            mask_sum = data.mask[i].sum()
            if abs(mask_sum - length) > 1e-6:
                validation_results['warnings'].append(f"Mask inconsistency for patient {i}: mask_sum={mask_sum}, length={length}")
                
        validation_results['valid'] = len(validation_results['errors']) == 0
        
    except Exception as e:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Validation failed with exception: {str(e)}")
        
    return validation_results


if __name__ == "__main__":
    # Add validation to the main execution
    data = build_trajectory_graph(
        traj_csv="Input/processed/trajectory_vent.csv",
        comorbidity_csv="Input/processed/aki_comorbidity_filtered.csv",
        action_cols=['peep_bin', 'fio2_bin', 'tidal_volume_ibw_bin'],
        k=3, 
        T_max=30,
        save_path="Input/processed/patient_traj_graph_vent.pt",
        mortality_csv="Input/raw/shared/aki_90day_mortality.csv"
    )
    
    # Validate the constructed graph
    validation_results = validate_trajectory_graph_integrity(data)
    
    print("\n" + "="*60)
    print("TRAJECTORY GRAPH VALIDATION REPORT")
    print("="*60)
    print(f"Valid: {'✓' if validation_results['valid'] else '✗'}")
    
    if validation_results['errors']:
        print(f"\nErrors ({len(validation_results['errors'])}):")
        for error in validation_results['errors']:
            print(f"  ❌ {error}")
            
    if validation_results['warnings']:
        print(f"\nWarnings ({len(validation_results['warnings'])}):")
        for warning in validation_results['warnings']:
            print(f"  ⚠️  {warning}")
            
    print(f"\nStatistics:")
    for key, value in validation_results['statistics'].items():
        print(f"  {key}: {value}")
    
    print("="*60) 