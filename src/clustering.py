# src/cluster.py
from __future__ import annotations
from typing import Dict, Iterable, List
import numpy as np
import pandas as pd

try:
    # nice progress bar if available
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k): return x  # fallback: no progress bar


def add_synapse_counts(calculation_nodes: pd.DataFrame,
                       node_counts: pd.Series) -> pd.DataFrame:
    """
    Add 'synapse_count' to calculation_nodes from node_counts (Series indexed by node_id).
    Returns a new DataFrame (copy).
    """
    df = calculation_nodes.copy()
    df["synapse_count"] = (
        df["node_id"].map(node_counts).fillna(0).astype(int)
    )
    return df


def _build_density_lookup(calculation_nodes: pd.DataFrame) -> Dict[int, float]:
    """
    Fast lookup: node_id -> synapse_density
    Requires columns: ['node_id', 'synapse_density'].
    """
    s = calculation_nodes.set_index("node_id")["synapse_density"]
    return s.to_dict()


def get_local_maximum(neuron_skel,
                      calculation_nodes: pd.DataFrame,
                      synapse_nodes: Iterable[int]) -> Dict[int, int]:
    """
    EXACT behavior of your notebook:
    For each start node in synapse_nodes, repeatedly move to the neighbor with the
    HIGHEST density if that neighbor's density is strictly greater than current.
    Stop when no neighbor is strictly higher → local maximum (peak).

    Returns: dict { start_node_id -> peak_node_id }.
    """
    # navis skeleton graph; keep it undirected like your code
    G = neuron_skel.graph.to_undirected()
    node2density = _build_density_lookup(calculation_nodes)

    peak_cache: Dict[int, int] = {}

    def climb_to_peak(start: int) -> int:
        if start in peak_cache:
            return peak_cache[start]

        current = start
        while True:
            curr_dens = node2density[current]
            neighbors = list(G.neighbors(current))
            if not neighbors:
                peak_cache[current] = current
                return current

            # densities of neighbors
            nbr_dens = np.fromiter((node2density[n] for n in neighbors), dtype=float, count=len(neighbors))
            diff = nbr_dens - curr_dens

            # no strictly-higher neighbor → local max
            if np.all(diff <= 0):
                peak_cache[current] = current
                return current

            # pick neighbor with largest positive difference (same as argmax in your code)
            best_idx = int(diff.argmax())
            best_neighbor = int(neighbors[best_idx])

            # short-circuit if we already know that neighbor's peak
            if best_neighbor in peak_cache:
                peak_cache[current] = peak_cache[best_neighbor]
                return peak_cache[best_neighbor]
            else:
                current = best_neighbor

    result: Dict[int, int] = {}
    for node in tqdm(synapse_nodes, desc="Nodes processed"):
        node = int(node)
        result[node] = int(climb_to_peak(node))
    return result


def group_by_peak(peaks: Dict[int, int]) -> Dict[int, List[int]]:
    """
    Convert {node -> peak} to {peak -> [nodes]}.
    """
    clusters: Dict[int, List[int]] = {}
    for node, peak in peaks.items():
        clusters.setdefault(int(peak), []).append(int(node))
    return clusters


def find_peak_by_iteration(node: int, clusters: Dict[int, List[int]]) -> int:
    """
    Returns the peak node (original cluster id) by iterating over clusters_dict.
    If no peak is found, returns -1.
    
    This matches the behavior from dc_initial_algo.py where each node is mapped
    to its corresponding peak node based on the clustering results.
    """
    for peak, nodes in clusters.items():
        if node in nodes:
            return peak
    return -1


def add_peak_node_to_calculation_nodes(calculation_nodes: pd.DataFrame, 
                                     clusters: Dict[int, List[int]]) -> pd.DataFrame:
    """
    Add 'peak_node' column to calculation_nodes DataFrame.
    Each node is mapped to its corresponding peak node from the clustering results.
    """
    df = calculation_nodes.copy()
    df["peak_node"] = df["node_id"].apply(lambda node: find_peak_by_iteration(node, clusters))
    return df


def summarize_clusters(clusters: Dict[int, List[int]]) -> dict:
    """
    Return summary stats like you print in the notebook.
    """
    largest_peak = None
    largest_count = 0
    total_nodes = 0

    for peak, nodes in clusters.items():
        n = len(nodes)
        total_nodes += n
        if n > largest_count:
            largest_count = n
            largest_peak = peak

    return {
        "n_clusters": len(clusters),
        "largest_peak": int(largest_peak) if largest_peak is not None else None,
        "largest_count": int(largest_count),
        "total_assigned_nodes": int(total_nodes),
    }
