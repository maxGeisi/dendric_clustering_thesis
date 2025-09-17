# src/processing/distance_analysis.py
"""
Distance analysis functions for inhibitory to excitatory synapse mapping.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import defaultdict


def find_closest_excitatory_synapses(
    syn_inh_df: pd.DataFrame,
    syn_exec_df: pd.DataFrame,
    geodesic_mat_full: pd.DataFrame
) -> pd.DataFrame:
    """
    Find the closest excitatory synapse for each inhibitory synapse.
    
    Args:
        syn_inh_df: DataFrame with inhibitory synapse data
        syn_exec_df: DataFrame with excitatory synapse data
        geodesic_mat_full: Geodesic distance matrix between all nodes
        
    Returns:
        DataFrame with inhibitory synapses mapped to closest excitatory synapses
    """
    syn_inh_df = syn_inh_df.copy()
    
    # Get excitatory synapse data
    e_ids = syn_exec_df["id"].to_numpy()
    e_nodes = syn_exec_df["closest_node_id"].to_numpy()
    
    def find_true_closest_esyn(i_row):
        """Find the closest excitatory synapse to an inhibitory synapse."""
        i_node = i_row.closest_node_id
        
        # Vector of geodesic distances from this I-node to every E-node
        # geodesic_mat_full is a DataFrame; .loc returns a Series aligned on e_nodes
        geo_to_all_e = geodesic_mat_full.loc[i_node, e_nodes].to_numpy()
        
        # Total per-E-synapse distance
        total_dists = geo_to_all_e
        
        # Pick the index of the minimum
        idx = np.argmin(total_dists)
        
        return pd.Series({
            "closest_e_syn_id": e_ids[idx],
            "min_dist_true_e": total_dists[idx]
        })
    
    # Apply mapping to all inhibitory synapses
    syn_inh_df[["closest_e_syn_id", "min_dist_e_syn_tot"]] = syn_inh_df.apply(find_true_closest_esyn, axis=1)
    
    return syn_inh_df


def map_inhibitory_to_excitatory_clusters_by_distance(
    syn_inh_df: pd.DataFrame,
    syn_exec_df: pd.DataFrame,
    cluster_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Map inhibitory synapses to excitatory clusters based on closest excitatory synapse.
    
    Args:
        syn_inh_df: DataFrame with inhibitory synapse data (must have closest_e_syn_id)
        syn_exec_df: DataFrame with excitatory synapse data
        cluster_df: DataFrame with excitatory cluster data
        
    Returns:
        DataFrame with inhibitory synapses mapped to excitatory clusters
    """
    syn_inh_df = syn_inh_df.copy()
    
    # Set of E-clusters that survived filtering
    valid_exec_ids = set(cluster_df["cluster_id"])
    
    # Create mapping from excitatory synapse ID to cluster ID
    exec_cluster_map = syn_exec_df.set_index("id")["cluster_id"].to_dict()
    
    # Map every I-synapse to the E-cluster of its closest E-synapse
    syn_inh_df["cluster_id_exec"] = (
        syn_inh_df["closest_e_syn_id"]
        .map(exec_cluster_map)        # id → cluster_id_exec or NaN
        .fillna(-1)                   # anything missing → –1
        .astype(int)
    )
    
    # Drop mappings onto filtered-out E-clusters:
    syn_inh_df.loc[
        ~syn_inh_df["cluster_id_exec"].isin(valid_exec_ids),
        "cluster_id_exec"
    ] = -1
    
    return syn_inh_df


def split_mixed_inhibitory_clusters_by_distance(
    syn_inh_df: pd.DataFrame,
    valid_exec_ids: set
) -> pd.DataFrame:
    """
    Split mixed inhibitory clusters that span multiple excitatory clusters.
    
    Args:
        syn_inh_df: DataFrame with inhibitory synapse data (must have cluster_id and cluster_id_exec)
        valid_exec_ids: Set of valid excitatory cluster IDs
        
    Returns:
        DataFrame with split inhibitory clusters
    """
    syn_inh_df = syn_inh_df.copy()
    
    # Find those I-clusters that still span >1 E-cluster
    mixed_inh = (
        syn_inh_df[syn_inh_df["cluster_id_exec"] != -1]
        .groupby("cluster_id")["cluster_id_exec"]
        .nunique()
        .loc[lambda x: x > 1]
        .index
        .tolist()
    )
    
    # Split each mixed I-cluster into subclusters by combining IDs
    # e.g. original I cluster 17 mapped to E cluster 42 → new ID 17042
    def _split_inh(row):
        inh = int(row["cluster_id"])
        exec_ = int(row["cluster_id_exec"])
        if inh in mixed_inh:
            return inh * 1000 + exec_
        else:
            return inh
    
    # Keep the old I-cluster ID around
    syn_inh_df["cluster_id_inh_old"] = syn_inh_df["cluster_id"]
    
    # Overwrite so each synapse now has a unique I-cluster ID
    syn_inh_df["cluster_id_inh"] = syn_inh_df.apply(_split_inh, axis=1)
    
    # Sanity check: no I-cluster should still map to >1 E-cluster
    verify = (
        syn_inh_df[syn_inh_df["cluster_id_exec"] != -1]
        .groupby("cluster_id_inh")["cluster_id_exec"]
        .nunique()
    )
    assert all(verify <= 1), f"Still mixed: {verify[verify>1].index.tolist()}"
    
    return syn_inh_df


def compute_distances_within_clusters(
    syn_inh_df: pd.DataFrame,
    syn_exec_df: pd.DataFrame,
    geodesic_mat_full: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute minimum distances from inhibitory synapses to excitatory synapses within the same cluster.
    
    Args:
        syn_inh_df: DataFrame with inhibitory synapse data (must have cluster_id_exec)
        syn_exec_df: DataFrame with excitatory synapse data
        geodesic_mat_full: Geodesic distance matrix between all nodes
        
    Returns:
        DataFrame with distance information added
    """
    syn_inh_df = syn_inh_df.copy()
    
    # Build lookup: for each E-cluster, list its (syn_id, node, offset)
    exec_syn_2_clu_dict = defaultdict(list)
    for row in syn_exec_df.itertuples(index=False):
        cid = row.cluster_id
        exec_syn_2_clu_dict[cid].append((row.id,  # Using 'id' instead of 'Exec_syn_id'
                                        row.closest_node_id,
                                        0.0))  # No offset in our current data
    
    # Define function that returns NaN if no match, else the min-distance
    def get_min_dist_to_e(i_row):
        """Get minimum distance from inhibitory synapse to excitatory synapses in same cluster."""
        e_cid = i_row.cluster_id_exec
        # If no such cluster in the lookup, bail out with NaN
        if e_cid not in exec_syn_2_clu_dict:
            return np.nan
        
        i_node = i_row.closest_node_id
        best = np.inf
        
        for e_id, e_node, e_offset in exec_syn_2_clu_dict[e_cid]:
            d = geodesic_mat_full.loc[i_node, e_node]
            if d < best:
                best = d
        return best
    
    # Apply it to every row of syn_inh_df
    syn_inh_df["min_dist_e_syn_in_clu"] = syn_inh_df.apply(get_min_dist_to_e, axis=1)
    
    return syn_inh_df


def analyze_e_i_relationships(syn_inh_cluster_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze consistent vs inconsistent E/I groups.
    
    Args:
        syn_inh_cluster_df: DataFrame with inhibitory synapses that have valid E-cluster mapping
        
    Returns:
        Dictionary with relationship statistics
    """
    # Group by excitatory cluster and check inhibitory cluster uniqueness
    # Use cluster_id_inh if it exists, otherwise use cluster_id
    inh_col = 'cluster_id_inh' if 'cluster_id_inh' in syn_inh_cluster_df.columns else 'cluster_id'
    same_inh_per_exec = syn_inh_cluster_df.groupby('cluster_id_exec')[inh_col].nunique()
    
    # Get consistent and inconsistent groups
    consistent_groups = same_inh_per_exec[same_inh_per_exec == 1].index
    inconsistent_groups = same_inh_per_exec[same_inh_per_exec > 1].index
    
    # Compute relationship statistics
    relationship_stats = {
        'n_consistent_groups': len(consistent_groups),
        'n_inconsistent_groups': len(inconsistent_groups),
        'max_i_clusters_per_e_cluster': same_inh_per_exec.max(),
        'total_mapped_inh_synapses': len(syn_inh_cluster_df),
        'consistent_groups': consistent_groups.tolist(),
        'inconsistent_groups': inconsistent_groups.tolist()
    }
    
    return relationship_stats


def compute_distance_statistics(syn_inh_cluster_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive distance statistics for inhibitory synapses.
    
    Args:
        syn_inh_cluster_df: DataFrame with inhibitory synapses that have valid distances
        
    Returns:
        Dictionary with distance statistics
    """
    valid_distances = syn_inh_cluster_df["min_dist_e_syn_in_clu"].dropna()
    
    if len(valid_distances) == 0:
        return {
            'total_synapses': 0,
            'mean_distance': np.nan,
            'median_distance': np.nan,
            'max_distance': np.nan,
            'min_distance': np.nan,
            'std_distance': np.nan
        }
    
    distance_stats = {
        'total_synapses': len(valid_distances),
        'mean_distance': valid_distances.mean(),
        'median_distance': valid_distances.median(),
        'max_distance': valid_distances.max(),
        'min_distance': valid_distances.min(),
        'std_distance': valid_distances.std()
    }
    
    return distance_stats


def define_cutoff_strategies(distance_stats: Dict[str, Any]) -> Dict[str, float]:
    """
    Define multiple cutoff strategies based on distance statistics.
    
    Args:
        distance_stats: Dictionary with distance statistics
        
    Returns:
        Dictionary with cutoff strategies and their thresholds
    """
    if distance_stats['total_synapses'] == 0:
        return {}
    
    mean_dist = distance_stats['mean_distance']
    median_dist = distance_stats['median_distance']
    std_dist = distance_stats['std_distance']
    
    cutoff_strategies = {
        "mean": mean_dist,
        "median": median_dist,
        "mean_plus_std": mean_dist + std_dist,
        "mean_plus_2std": mean_dist + 2 * std_dist,
        "percentile_75": np.percentile([mean_dist, median_dist], 75),  # Approximation
        "percentile_90": np.percentile([mean_dist, median_dist], 90),  # Approximation
        "percentile_95": np.percentile([mean_dist, median_dist], 95)   # Approximation
    }
    
    return cutoff_strategies


def apply_distance_cutoff(
    syn_inh_cluster_df: pd.DataFrame,
    cutoff_threshold: float
) -> pd.DataFrame:
    """
    Apply distance cutoff to filter inhibitory synapses.
    
    Args:
        syn_inh_cluster_df: DataFrame with inhibitory synapses
        cutoff_threshold: Distance threshold for filtering
        
    Returns:
        Filtered DataFrame
    """
    return syn_inh_cluster_df[
        syn_inh_cluster_df["min_dist_e_syn_in_clu"] <= cutoff_threshold
    ].copy()


def print_distance_analysis_summary(
    syn_inh_df: pd.DataFrame,
    syn_inh_cluster_df: pd.DataFrame,
    syn_inh_cluster_df_filtered: pd.DataFrame,
    distance_stats: Dict[str, Any],
    relationship_stats: Dict[str, Any],
    cutoff_strategy: str,
    cutoff_threshold: float
) -> None:
    """
    Print comprehensive summary of distance analysis results.
    
    Args:
        syn_inh_df: Original inhibitory synapse DataFrame
        syn_inh_cluster_df: Inhibitory synapses with valid E-mapping
        syn_inh_cluster_df_filtered: Inhibitory synapses after distance cutoff
        distance_stats: Distance statistics
        relationship_stats: E/I relationship statistics
        cutoff_strategy: Name of cutoff strategy used
        cutoff_threshold: Threshold value used
    """
    print("=" * 80)
    print("DISTANCE ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nDISTANCE STATISTICS:")
    print(f"   Total inhibitory synapses with valid distances: {distance_stats['total_synapses']}")
    print(f"   Mean distance to closest E-synapse: {distance_stats['mean_distance']:.3f} μm")
    print(f"   Median distance to closest E-synapse: {distance_stats['median_distance']:.3f} μm")
    print(f"   Max distance to closest E-synapse: {distance_stats['max_distance']:.3f} μm")
    print(f"   Min distance to closest E-synapse: {distance_stats['min_distance']:.3f} μm")
    print(f"   Std distance to closest E-synapse: {distance_stats['std_distance']:.3f} μm")
    
    print(f"\nFILTERING RESULTS:")
    print(f"   Original inhibitory synapses: {len(syn_inh_df)}")
    print(f"   Synapses with valid E-mapping: {len(syn_inh_cluster_df)}")
    print(f"   Synapses after distance cutoff: {len(syn_inh_cluster_df_filtered)}")
    if len(syn_inh_cluster_df) > 0:
        print(f"   Filtering efficiency: {len(syn_inh_cluster_df_filtered)/len(syn_inh_cluster_df)*100:.1f}%")
    
    print(f"\nE/I RELATIONSHIP ANALYSIS:")
    print(f"   Consistent E/I groups: {relationship_stats['n_consistent_groups']}")
    print(f"   Inconsistent E/I groups: {relationship_stats['n_inconsistent_groups']}")
    print(f"   Max I-clusters per E-cluster: {relationship_stats['max_i_clusters_per_e_cluster']}")
    print(f"   Mapped inhibitory synapses: {relationship_stats['total_mapped_inh_synapses']}")
    
    print(f"\nCUTOFF STRATEGY:")
    print(f"   Strategy: {cutoff_strategy}")
    print(f"   Threshold: {cutoff_threshold:.3f} μm")
    
    print("=" * 80)
