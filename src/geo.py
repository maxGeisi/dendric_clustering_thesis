# src/geo.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import navis
import joblib
import os
import tempfile
import pickle

# ---------- utils ----------

def ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def _atomic_dump(obj, path: Path):
    """Write with a temp file then atomic rename to avoid truncated cache files."""
    path = Path(path)
    tmpdir = path.parent
    ensure_dir(tmpdir)
    with tempfile.NamedTemporaryFile(dir=tmpdir, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        joblib.dump(obj, tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX
    finally:
        if tmp_path.exists():
            try: tmp_path.unlink()
            except Exception: pass

def _safe_load_or_none(path: Path):
    """Return loaded object, or None if the cache is unreadable/corrupted."""
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Cache unreadable ({e.__class__.__name__}): {path}. Deleting and recomputing.")
        try: Path(path).unlink()
        except FileNotFoundError:
            pass
        return None

# ---------- cache paths ----------

def geodesic_cache_path(interim_dir: str | Path, neuron_id: str) -> Path:
    return Path(interim_dir) / f"geodesic_mat_{neuron_id}.pkl"

def kernel_cache_path(interim_dir: str | Path, neuron_id: str, sigma_nm: float) -> Path:
    """Generate cache path matching the existing naming convention: node_weight_mat_n3_sigma:1.0.pkl"""
    return Path(interim_dir) / f"node_weight_mat_{neuron_id}_sigma:{sigma_nm}.pkl"

# ---------- public API ----------

def compute_or_load_geodesic(
    neuron_skel,
    interim_dir: str | Path,
    neuron_id: str,
    force_recompute: bool = False,
):
    """
    Returns geodesic matrix (pandas.DataFrame) with node indices/cols.
    Caches to joblib pickle under interim_dir. Auto-heals corrupted cache.
    """
    interim_dir = ensure_dir(interim_dir)
    pkl = geodesic_cache_path(interim_dir, neuron_id)

    g = None
    if (not force_recompute) and Path(pkl).exists():
        print("Found geodesic cache:", pkl, "→ loading")
        g = _safe_load_or_none(pkl)

    if g is None:
        print("Computing geodesic matrix …")
        g = navis.geodesic_matrix(neuron_skel, weight="weight")
        print("Geodesic matrix shape:", g.shape)
        _atomic_dump(g, pkl)
        print("Saved:", pkl)

    # sanity
    arr = g.values if isinstance(g, pd.DataFrame) else g
    if np.isinf(arr).any():
        print("Geodesic matrix contains ∞ values.")
    return g


def compute_or_load_node_kernel(
    geodesic_mat,
    interim_dir: str | Path,
    neuron_id: str,
    sigma_nm: float,
    force_recompute: bool = False,
):
    """
    Build W = exp(-d^2/(2σ^2)) on node geodesic distances (nm).
    Normalize by σ*sqrt(2π) → W_norm.
    Returns (W_nodes, W_nodes_normalized) as DataFrames if geodesic_mat is a DataFrame.

    Uses the exact same logic as the original dc_initial_algo.py file.
    """
    interim_dir = ensure_dir(interim_dir)
    pkl = kernel_cache_path(interim_dir, neuron_id, sigma_nm)

    W_nodes = None
    if (not force_recompute) and Path(pkl).exists():
        print(f"Found file: {pkl}. Loading node weight matrix...")
        with open(pkl, 'rb') as f:
            W_nodes = pickle.load(f)

    if W_nodes is None:
        print("File does not exist. Computing weight matrix...")
        # W = np.exp(-synapse_geodesic_distances**2 / (2.0 * sigma**2))
        W_nodes = np.exp(-geodesic_mat**2 / (2.0 * sigma_nm**2))

        # Save the computed weight matrix to a file for future use
        with open(pkl, 'wb') as f:
            pickle.dump(W_nodes, f)
        print(f"Node weight matrix saved to: {pkl}")

    # Now compute the normalized matrix 
    integral_factor = sigma_nm * np.sqrt(2.0 * np.pi)  
    W_nodes_normalized = W_nodes / integral_factor
    
    return W_nodes, W_nodes_normalized
