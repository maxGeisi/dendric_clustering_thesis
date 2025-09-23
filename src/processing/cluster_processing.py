# src/processing/cluster_processing.py
"""
Cluster processing functions for dendric clustering analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Any

def build_cluster_dataframe(
    syn_df: pd.DataFrame,
    clusters_dict: Dict[int, List[int]],
    peak_to_cluster_id: Dict[int, int],
    new_cluster_id_to_peak: Dict[int, int],
    syn_id_col: str = "id",
    cluster_id_col: str = "cluster_id",
    synapse_type: str = "excitatory"
) -> pd.DataFrame:
    """
    Build cluster DataFrame from synapse data and clustering results.
    
    Args:
        syn_df: DataFrame with synapse data and cluster assignments
        clusters_dict: Dictionary mapping peak nodes to lists of nodes
        peak_to_cluster_id: Mapping from peak nodes to cluster IDs
        new_cluster_id_to_peak: Reverse mapping from cluster IDs to peak nodes
        syn_id_col: Column name for synapse IDs
        cluster_id_col: Column name for cluster IDs (e.g., "cluster_id" or "cluster_id_inh")
        synapse_type: Type of synapses ("excitatory" or "inhibitory")
        
    Returns:
        DataFrame with cluster information
    """
    # Group synapse data by cluster_id and aggregate
    cluster_df = syn_df.groupby(cluster_id_col).agg(
        Synapse_Count=(cluster_id_col, "size"),
        Synapses=(syn_id_col, lambda x: list(x))
    ).reset_index()
    
    # Rename the cluster_id column to standardize
    cluster_df = cluster_df.rename(columns={cluster_id_col: "cluster_id"})
    
    # Rename Synapse_Count based on synapse type
    if synapse_type == "inhibitory":
        cluster_df = cluster_df.rename(columns={"Synapse_Count": "i_synapse_count"})
    else:  # excitatory
        cluster_df = cluster_df.rename(columns={"Synapse_Count": "e_synapse_count"})
    
    # Add peak information
    cluster_df["Cluster_Peak"] = cluster_df["cluster_id"].map(new_cluster_id_to_peak)
    
    # Add associated nodes for each peak
    def find_nodes_by_peak(peak_id):
        if peak_id in clusters_dict:
            return clusters_dict[peak_id]
        return []
    
    cluster_df["Associated_Nodes"] = cluster_df["Cluster_Peak"].apply(find_nodes_by_peak)
    
    # Reorder columns based on synapse type
    if synapse_type == "inhibitory":
        cluster_df = cluster_df[["cluster_id", "Cluster_Peak", "i_synapse_count", "Synapses", "Associated_Nodes"]]
    else:  # excitatory
        cluster_df = cluster_df[["cluster_id", "Cluster_Peak", "e_synapse_count", "Synapses", "Associated_Nodes"]]
    
    return cluster_df

def compute_cluster_metrics(
    cluster_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    geodesic_mat_full: pd.DataFrame,
    neuron_skel
) -> pd.DataFrame:
    """
    Compute additional metrics for clusters (cable length, density, max distance).
    
    Args:
        cluster_df: DataFrame with cluster information
        syn_df: DataFrame with synapse data
        geodesic_mat_full: Geodesic distance matrix
        neuron_skel: Navis neuron skeleton object
        
    Returns:
        DataFrame with added cluster metrics
    """
    cluster_df = cluster_df.copy()
    
    def compute_minimal_cable_length(cluster_id):
        """Compute minimal cable length for cluster nodes."""
        cluster_nodes = cluster_df[cluster_df["cluster_id"] == cluster_id]["Associated_Nodes"].iloc[0]
        if len(cluster_nodes) < 2:
            return 0.0
        
        # Extract submatrix for cluster nodes
        node_subset = [n for n in cluster_nodes if n in geodesic_mat_full.index]
        if len(node_subset) < 2:
            return 0.0
        
        submatrix = geodesic_mat_full.loc[node_subset, node_subset]
        
        # Find minimum spanning tree
        from scipy.sparse.csgraph import minimum_spanning_tree
        mst = minimum_spanning_tree(submatrix)
        return mst.sum()
    
    def compute_cluster_density(synapse_count, cable_length):
        """Compute synapse density (synapses per unit length)."""
        if cable_length == 0:
            return 0.0
        return synapse_count / cable_length
    
    
    # Compute metrics
    cluster_df["minimal_cable_length"] = cluster_df["cluster_id"].apply(compute_minimal_cable_length)
    # Determine synapse count column name
    synapse_count_col = "e_synapse_count" if "e_synapse_count" in cluster_df.columns else "i_synapse_count"
    
    cluster_df["cluster_density"] = cluster_df.apply(
        lambda row: compute_cluster_density(row[synapse_count_col], row["minimal_cable_length"]), 
        axis=1
    )
    
    return cluster_df

def filter_clusters_by_density(
    cluster_df: pd.DataFrame,
    overall_density: float,
    min_density_factor: float = 1.0
) -> pd.DataFrame:
    """
    Filter clusters based on density threshold.
    
    Args:
        cluster_df: DataFrame with cluster information
        overall_density: Overall synapse density of the neuron
        min_density_factor: Minimum density factor relative to overall density
        
    Returns:
        Filtered DataFrame
    """
    threshold = overall_density * min_density_factor
    filtered_df = cluster_df[cluster_df["cluster_density"] > threshold].copy()
    
    print(f"Filtered clusters: {len(cluster_df)} -> {len(filtered_df)}")
    print(f"Density threshold: {threshold:.5f}")
    
    return filtered_df

def mark_synapses_in_valid_clusters(
    syn_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    cluster_id_col: str = "cluster_id"
) -> pd.DataFrame:
    """
    Mark which synapses remain in valid (non-filtered) clusters.
    
    Args:
        syn_df: DataFrame with synapse data and cluster assignments
        cluster_df: DataFrame with valid clusters (after filtering)
        cluster_id_col: Column name for cluster IDs
        
    Returns:
        DataFrame with added 'in_valid_cluster' column
    """
    syn_df = syn_df.copy()
    
    # Get set of valid cluster IDs
    valid_cluster_ids = set(cluster_df["cluster_id"])
    
    # Mark synapses that are in valid clusters
    syn_df["in_valid_cluster"] = syn_df[cluster_id_col].isin(valid_cluster_ids)
    
    return syn_df

def add_inhibitory_synapse_counts(
    cluster_df: pd.DataFrame,
    syn_inh_cluster_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add inhibitory synapse count and has_I_associated properties to cluster_df.
    
    Args:
        cluster_df: Excitatory cluster DataFrame
        syn_inh_cluster_df: Inhibitory synapse DataFrame with cluster_id_exec column
        
    Returns:
        DataFrame with added i_synapse_count and has_I_associated columns
    """
    cluster_df = cluster_df.copy()
    
    # Count inhibitory synapses per excitatory cluster
    syn_counts = (
        syn_inh_cluster_df
        .groupby("cluster_id_exec")
        .size()             # Series: index=cluster_id_exec, value=number of rows
        .to_dict()          # { exec_id: count, … }
    )
    
    # Map those counts back onto cluster_df.cluster_id
    cluster_df["i_synapse_count"] = (
        cluster_df["cluster_id"]
        .map(syn_counts)    # gives NaN for clusters with 0 entries
        .fillna(0)          # turn NaN → 0
        .astype(int)        # make them ints again
    )
    
    # Add has_I_associated column
    cluster_df["has_I_associated"] = cluster_df["i_synapse_count"] > 0
    
    return cluster_df

def add_inhibitory_cluster_counts(
    cluster_df: pd.DataFrame,
    syn_inh_cluster_df_final: pd.DataFrame
) -> pd.DataFrame:
    """
    Add number of assigned inhibitory clusters to each excitatory cluster.
    
    Args:
        cluster_df: Excitatory cluster DataFrame
        syn_inh_cluster_df_final: Final inhibitory cluster DataFrame after filtering
        
    Returns:
        Updated cluster_df with num_assigned_i_cluster column
    """
    cluster_df = cluster_df.copy()
    
    # Create mapping from E-cluster ID to number of I-clusters
    same_inh_per_exec = {}
    
    for _, row in syn_inh_cluster_df_final.iterrows():
        e_cluster_id = row.get('cluster_id_exec', -1)
        if e_cluster_id != -1:  # Valid mapping
            if e_cluster_id not in same_inh_per_exec:
                same_inh_per_exec[e_cluster_id] = 0
            same_inh_per_exec[e_cluster_id] += 1
    
    def assign_number_of_i_clu_to_e_clu(row):
        row_id = row.cluster_id
        for e_id, count in same_inh_per_exec.items():
            if e_id == row_id:
                return count
        return 0
    
    cluster_df["num_assigned_i_cluster"] = cluster_df.apply(assign_number_of_i_clu_to_e_clu, axis=1)
    
    return cluster_df


def print_cluster_statistics(cluster_df: pd.DataFrame) -> None:
    """
    Print comprehensive statistics for cluster properties.
    
    Args:
        cluster_df: DataFrame with cluster information including metrics
    """
    if len(cluster_df) == 0:
        print("No clusters to analyze.")
        return
    
    print("\n" + "="*60)
    print("CLUSTER STATISTICS")
    print("="*60)
    
    # Determine synapse count column name and label
    if "e_synapse_count" in cluster_df.columns:
        syn_counts = cluster_df["e_synapse_count"]
        syn_label = "SYNAPSE COUNT"
    elif "i_synapse_count" in cluster_df.columns:
        syn_counts = cluster_df["i_synapse_count"]
        syn_label = "INHIBITORY SYNAPSE COUNT"
    else:
        # Fallback for backward compatibility
        syn_counts = cluster_df["Synapse_Count"]
        syn_label = "SYNAPSE COUNT"
    
    print(f"\n{syn_label}:")
    print(f"  Max  # syn:   {syn_counts.max():.0f}")
    print(f"  Mean # syn:   {syn_counts.mean():.2f}")
    print(f"  Median # syn: {syn_counts.median():.1f}")
    print(f"  Min  # syn:   {syn_counts.min():.0f}")
    print(f"  Std  # syn:   {syn_counts.std():.2f}")
    
    # Cluster Density Statistics
    densities = cluster_df["cluster_density"]
    density_label = "CLUSTER DENSITY" if "e_synapse_count" in cluster_df.columns else "INHIBITORY CLUSTER DENSITY"
    print(f"\n{density_label} (synapses/μm):")
    print(f"  Max  density:   {densities.max():.3f}")
    print(f"  Mean density:   {densities.mean():.3f}")
    print(f"  Median density: {densities.median():.3f}")
    print(f"  Min  density:   {densities.min():.3f}")
    print(f"  Std  density:   {densities.std():.3f}")
    
    # Minimal Cable Length Statistics
    cable_lengths = cluster_df["minimal_cable_length"]
    cable_label = "MINIMAL CABLE LENGTH" if "e_synapse_count" in cluster_df.columns else "INHIBITORY MINIMAL CABLE LENGTH"
    print(f"\n{cable_label} (μm):")
    print(f"  Max  length:   {cable_lengths.max():.3f}")
    print(f"  Mean length:   {cable_lengths.mean():.3f}")
    print(f"  Median length: {cable_lengths.median():.3f}")
    print(f"  Min  length:   {cable_lengths.min():.3f}")
    print(f"  Std  length:   {cable_lengths.std():.3f}")
    
    # Additional Summary Statistics
    print(f"\nSUMMARY:")
    print(f"  Total clusters: {len(cluster_df)}")
    print(f"  Total synapses: {syn_counts.sum()}")
    print(f"  Total cable length: {cable_lengths.sum():.2f} μm")
    print(f"  Overall density: {syn_counts.sum() / cable_lengths.sum():.3f} synapses/μm")
    
    print("="*60)


def compute_separated_cluster_statistics(cluster_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute cluster statistics separated by whether E clusters have I synapses associated with them.
    
    Args:
        cluster_df: DataFrame with columns 'e_synapse_count', 'minimal_cable_length', 
                   'cluster_density', 'i_synapse_count'
    
    Returns:
        Dictionary with statistics for clusters with and without I synapses
    """
    # Separate clusters based on I synapse association
    clusters_with_i = cluster_df[cluster_df['i_synapse_count'] > 0]
    clusters_without_i = cluster_df[cluster_df['i_synapse_count'] == 0]
    
    def compute_stats(subset_df):
        if len(subset_df) == 0:
            return {
                'count': 0,
                'e_synapse_count': {'mean': 0, 'median': 0, 'max': 0},
                'minimal_cable_length': {'mean': 0, 'median': 0, 'max': 0},
                'cluster_density': {'mean': 0, 'median': 0, 'max': 0}
            }
        
        return {
            'count': len(subset_df),
            'e_synapse_count': {
                'mean': subset_df['e_synapse_count'].mean(),
                'median': subset_df['e_synapse_count'].median(),
                'max': subset_df['e_synapse_count'].max()
            },
            'minimal_cable_length': {
                'mean': subset_df['minimal_cable_length'].mean(),
                'median': subset_df['minimal_cable_length'].median(),
                'max': subset_df['minimal_cable_length'].max()
            },
            'cluster_density': {
                'mean': subset_df['cluster_density'].mean(),
                'median': subset_df['cluster_density'].median(),
                'max': subset_df['cluster_density'].max()
            }
        }
    
    return {
        'with_i_synapses': compute_stats(clusters_with_i),
        'without_i_synapses': compute_stats(clusters_without_i)
    }


def compute_intra_cluster_distances(
    cluster_df: pd.DataFrame,
    syn_exec_df: pd.DataFrame,
    geodesic_mat_full: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute intra-cluster distance statistics for each cluster.
    
    Args:
        cluster_df: DataFrame with cluster information
        syn_exec_df: DataFrame with excitatory synapse data
        geodesic_mat_full: Geodesic distance matrix between all nodes
        
    Returns:
        DataFrame with added intra-cluster distance metrics
    """
    syn_exec_df = syn_exec_df.rename(columns={'cluster_id_exec':'e_cluster_id'})
    intra_stats = []
    for cid, grp in syn_exec_df.groupby('e_cluster_id'):
        node_ids = grp['closest_node_id'].unique()
        if len(node_ids) < 2:
            vals = [0., 0., 0., 0.]
        else:
            submat = geodesic_mat_full.loc[node_ids, node_ids].values
            triu   = submat[np.triu_indices(len(node_ids), k=1)]
            vals   = [triu.mean(), np.median(triu), triu.min(), triu.max()]
        intra_stats.append(
            dict(
                e_cluster_id=cid,
                mean_intra_dist=vals[0],
                median_intra_dist=vals[1],
                min_intra_dist=vals[2],
                max_intra_dist=vals[3]
            )
        )

    intra_df = pd.DataFrame(intra_stats)

    # 2) Now merge once, with no overlapping column names
    cluster_df = cluster_df.merge(intra_df, on='e_cluster_id', how='left')
    
    return cluster_df


def compute_inter_cluster_distances(
    cluster_df: pd.DataFrame,
    syn_exec_df: pd.DataFrame,
    geodesic_mat_full: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute inter-cluster distance statistics (median distance to nearest neighbor cluster).
    
    Args:
        cluster_df: DataFrame with cluster information
        syn_exec_df: DataFrame with excitatory synapse data
        geodesic_mat_full: Geodesic distance matrix between all nodes
        
    Returns:
        DataFrame with added inter-cluster distance metrics
    """
    syn_exec_df = syn_exec_df.rename(columns={'cluster_id_exec':'e_cluster_id'})
    inter_stats = []
    for cid, grp in syn_exec_df.groupby('e_cluster_id'):
        node_ids = grp['closest_node_id'].unique()
        # pull the full distance‐matrix rows for those synapses
        subdf = geodesic_mat_full.loc[node_ids, :].copy()
        # zero‐out / inf‐out any distance to a synapse *in the same cluster*
        subdf[node_ids] = np.inf
        # now each row’s min is the distance to its nearest synapse in *another* cluster
        nearest_out = subdf.min(axis=1)
        inter_stats.append({
            'e_cluster_id'        : cid,
            'inter_mean_dist'   : nearest_out.mean(),
            'inter_median_dist' : nearest_out.median(),
            'inter_min_dist'    : nearest_out.min(),
            'inter_max_dist'    : nearest_out.max(),
        })

    inter_df   = pd.DataFrame(inter_stats)
    cluster_df = cluster_df.merge(inter_df, on='e_cluster_id', how='left')
    
    return cluster_df
    
    


def print_separated_cluster_summary(stats: Dict[str, Dict[str, float]]) -> None:
    """
    Print a compact summary of cluster statistics separated by I synapse association.
    
    Args:
        stats: Dictionary returned by compute_separated_cluster_statistics
    """
    print("=" * 80)
    print("COMPACT CLUSTER SUMMARY (SEPARATED BY I SYNAPSE ASSOCIATION)")
    print("=" * 80)
    
    for category, data in stats.items():
        if data['count'] == 0:
            continue
            
        category_name = "E CLUSTERS WITH I SYNAPSES" if category == 'with_i_synapses' else "E CLUSTERS WITHOUT I SYNAPSES"
        print(f"\n{category_name} (n={data['count']}):")
        
        print(f"  CLUSTER SIZE (e_synapse_count):")
        print(f"    Mean: {data['e_synapse_count']['mean']:.1f} | Median: {data['e_synapse_count']['median']:.1f} | Max: {data['e_synapse_count']['max']}")
        
        print(f"  CABLE LENGTH (minimal_cable_length):")
        print(f"    Mean: {data['minimal_cable_length']['mean']:.2f} μm | Median: {data['minimal_cable_length']['median']:.2f} μm | Max: {data['minimal_cable_length']['max']:.2f} μm")
        
        print(f"  CLUSTER DENSITY (cluster_density):")
        print(f"    Mean: {data['cluster_density']['mean']:.2f} | Median: {data['cluster_density']['median']:.2f} | Max: {data['cluster_density']['max']:.2f}")
    
    print("=" * 80)
