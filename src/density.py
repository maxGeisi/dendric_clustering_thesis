# src/density.py
from __future__ import annotations
import pandas as pd

def node_density_from_counts(W_nodes_normalized: pd.DataFrame,
                             node_counts: pd.Series) -> pd.Series:
    """
    W_norm (nodes x nodes) dot node_counts (nodes,) -> node_density (nodes,)
    Reindex counts to W index; fill missing with 0.
    """
    counts = node_counts.reindex(W_nodes_normalized.index, fill_value=0)
    # pandas DataFrame.dot(Series) returns Series aligned on columns/index
    dens = W_nodes_normalized.dot(counts)
    return dens

def print_density_statistics(node_density: pd.Series) -> None:
    """
    Print density statistics matching the original dc_initial_algo.py format.
    """
    print(f"Maximum density: {node_density.max():.3f}")
    print(f"Average density: {node_density.mean():.3f}") 
    print(f"Minimum density: {node_density.min():.3f}")

def attach_density_to_nodes(neuron_skel, node_density: pd.Series) -> pd.DataFrame:
    """
    Return a copy of neuron_skel.nodes with 'synapse_density' column.
    Fill NaNs with the mean density and cast node_id to int.
    """
    nd = node_density.reindex(neuron_skel.nodes.index)  # align by node index
    nodes = neuron_skel.nodes.copy()
    nodes["synapse_density"] = nd
    nodes["synapse_density"] = nodes["synapse_density"].fillna(nodes["synapse_density"].mean())
    nodes["node_id"] = nodes["node_id"].astype(int)
    return nodes
