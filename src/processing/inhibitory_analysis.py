# src/processing/inhibitory_analysis.py
"""
Inhibitory synapse analysis functions for dendric clustering.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

def map_inhibitory_to_excitatory_clusters(
    syn_inh_df: pd.DataFrame,
    syn_exec_df: pd.DataFrame,
    clusters_dict_inh_e_gradient: Dict[int, List[int]],
    peak_to_cluster_id: Dict[int, int],
    valid_exec_ids: set
) -> pd.DataFrame:
    """
    Map inhibitory synapses to excitatory clusters using E-gradient approach.
    
    Args:
        syn_inh_df: DataFrame with inhibitory synapse data
        syn_exec_df: DataFrame with excitatory synapse data
        clusters_dict_inh_e_gradient: Clustering results for inhibitory synapses
        peak_to_cluster_id: Mapping from peaks to cluster IDs
        valid_exec_ids: Set of valid excitatory cluster IDs
        
    Returns:
        DataFrame with inhibitory synapses mapped to excitatory clusters
    """
    syn_inh_df = syn_inh_df.copy()
    
    def assign_e_cid_to_i_syn(node):
        """Assign excitatory cluster ID to inhibitory synapse based on node."""
        for peak, nodes in clusters_dict_inh_e_gradient.items():
            if node in nodes:
                if peak in peak_to_cluster_id:
                    wanted_cid = peak_to_cluster_id[peak]
                    if wanted_cid in valid_exec_ids:
                        return wanted_cid
        return -1
    
    # Map inhibitory synapses to excitatory clusters
    syn_inh_df["cluster_id_exec"] = syn_inh_df["closest_node_id"].apply(assign_e_cid_to_i_syn)
    
    return syn_inh_df

def split_mixed_inhibitory_clusters(
    syn_inh_df: pd.DataFrame,
    valid_exec_ids: set
) -> pd.DataFrame:
    """
    Split inhibitory clusters that map to multiple excitatory clusters.
    
    Args:
        syn_inh_df: DataFrame with inhibitory synapse data
        valid_exec_ids: Set of valid excitatory cluster IDs
        
    Returns:
        DataFrame with split inhibitory clusters
    """
    syn_inh_df = syn_inh_df.copy()
    
    # Find inhibitory clusters that map to multiple excitatory clusters
    mixed_inh = (
        syn_inh_df[syn_inh_df["cluster_id_exec"] != -1]
        .groupby("cluster_id_inh")["cluster_id_exec"]
        .nunique()
        .loc[lambda x: x > 1]
        .index
        .tolist()
    )
    
    print(f"Found {len(mixed_inh)} mixed inhibitory clusters: {mixed_inh[:10]}...")
    
    def split_inh_cluster(row):
        """Split mixed inhibitory clusters by combining IDs."""
        inh = int(row["cluster_id_inh"])
        exec_ = int(row["cluster_id_exec"])
        if inh in mixed_inh:
            return inh * 1000 + exec_
        else:
            return inh
    
    # Keep old cluster ID
    syn_inh_df["cluster_id_inh_old"] = syn_inh_df["cluster_id_inh"]
    
    # Apply splitting
    syn_inh_df["cluster_id_inh"] = syn_inh_df.apply(split_inh_cluster, axis=1)
    
    return syn_inh_df

def compute_e_i_relationships(
    syn_inh_df: pd.DataFrame,
    syn_exec_df: pd.DataFrame,
    cluster_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute E/I relationship statistics and create filtered datasets.
    
    Args:
        syn_inh_df: DataFrame with inhibitory synapse data
        syn_exec_df: DataFrame with excitatory synapse data
        cluster_df: DataFrame with cluster information
        
    Returns:
        Tuple of (filtered_inh_df, relationship_stats)
    """
    # Filter inhibitory synapses that could be assigned to excitatory clusters
    syn_inh_cluster_df = syn_inh_df[syn_inh_df["cluster_id_exec"] != -1].copy()
    
    # Group by excitatory cluster and check inhibitory cluster uniqueness
    same_inh_per_exec = syn_inh_cluster_df.groupby('cluster_id_exec')['cluster_id_inh'].nunique()
    
    # Get consistent and inconsistent groups
    consistent_groups = same_inh_per_exec[same_inh_per_exec == 1].index
    inconsistent_groups = same_inh_per_exec[same_inh_per_exec > 1].index
    
    # Compute relationship statistics
    relationship_stats = {
        'n_consistent_groups': len(consistent_groups),
        'n_inconsistent_groups': len(inconsistent_groups),
        'max_i_clusters_per_e_cluster': same_inh_per_exec.max(),
        'total_mapped_inh_synapses': len(syn_inh_cluster_df),
        'total_inh_synapses': len(syn_inh_df)
    }
    
    # Add inhibitory synapse counts to cluster DataFrame
    syn_counts = (
        syn_inh_cluster_df
        .groupby("cluster_id_exec")
        .size()
        .to_dict()
    )
    
    cluster_df["i_synapse_count"] = (
        cluster_df["cluster_id"]
        .map(syn_counts)
        .fillna(0)
        .astype(int)
    )
    
    cluster_df["has_I_associated"] = cluster_df["i_synapse_count"] > 0
    
    return syn_inh_cluster_df, relationship_stats

def compute_e_i_distance_analysis(
    syn_inh_cluster_df: pd.DataFrame,
    syn_exec_df: pd.DataFrame,
    geodesic_mat_full: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute geodesic distances between inhibitory and excitatory synapses within clusters.
    
    Args:
        syn_inh_cluster_df: DataFrame with inhibitory synapses mapped to E clusters
        syn_exec_df: DataFrame with excitatory synapse data
        geodesic_mat_full: Geodesic distance matrix between all nodes
        
    Returns:
        DataFrame with geodesic distance analysis
    """
    def get_min_geodesic_dist_to_e_synapse(inh_row):
        """Find minimum geodesic distance to excitatory synapse in same cluster."""
        cluster_id = inh_row["cluster_id_exec"]
        e_synapses_in_cluster = syn_exec_df[syn_exec_df["cluster_id"] == cluster_id]
        
        if len(e_synapses_in_cluster) == 0:
            return np.nan
        
        # Get the node that the inhibitory synapse is mapped to
        inh_node = inh_row["closest_node_id"]
        
        # Get nodes that excitatory synapses are mapped to
        e_nodes = e_synapses_in_cluster["closest_node_id"].values
        
        # Compute geodesic distances from inhibitory node to all excitatory nodes
        if inh_node in geodesic_mat_full.index:
            distances = geodesic_mat_full.loc[inh_node, e_nodes].values
            return np.min(distances)
        else:
            return np.nan
    
    syn_inh_cluster_df = syn_inh_cluster_df.copy()
    syn_inh_cluster_df["min_geodesic_dist_to_e_syn"] = syn_inh_cluster_df.apply(get_min_geodesic_dist_to_e_synapse, axis=1)
    
    return syn_inh_cluster_df


def print_inhibitory_cluster_statistics(syn_inh_cluster_df: pd.DataFrame) -> None:
    """
    Print comprehensive statistics for inhibitory cluster properties.
    
    Args:
        syn_inh_cluster_df: DataFrame with inhibitory cluster information
    """
    if len(syn_inh_cluster_df) == 0:
        print("No inhibitory clusters to analyze.")
        return
    
    print("\n" + "="*60)
    print("INHIBITORY CLUSTER STATISTICS")
    print("="*60)
    
    # Inhibitory Synapse Count Statistics
    if "i_synapse_count" in syn_inh_cluster_df.columns:
        syn_counts = syn_inh_cluster_df["i_synapse_count"]
    elif "Synapse_Count" in syn_inh_cluster_df.columns:
        syn_counts = syn_inh_cluster_df["Synapse_Count"]
    else:
        print("No synapse count column found")
        return
    
    print(f"\nINHIBITORY SYNAPSE COUNT:")
    print(f"  Max  # syn:   {syn_counts.max():.0f}")
    print(f"  Mean # syn:   {syn_counts.mean():.2f}")
    print(f"  Median # syn: {syn_counts.median():.1f}")
    print(f"  Min  # syn:   {syn_counts.min():.0f}")
    print(f"  Std  # syn:   {syn_counts.std():.2f}")
    
    # Cluster Density Statistics
    if "cluster_density" in syn_inh_cluster_df.columns:
        densities = syn_inh_cluster_df["cluster_density"]
        print(f"\nINHIBITORY CLUSTER DENSITY (synapses/μm):")
        print(f"  Max  density:   {densities.max():.3f}")
        print(f"  Mean density:   {densities.mean():.3f}")
        print(f"  Median density: {densities.median():.3f}")
        print(f"  Min  density:   {densities.min():.3f}")
        print(f"  Std  density:   {densities.std():.3f}")
    
    # Minimal Cable Length Statistics
    if "minimal_cable_length" in syn_inh_cluster_df.columns:
        cable_lengths = syn_inh_cluster_df["minimal_cable_length"]
        print(f"\nINHIBITORY MINIMAL CABLE LENGTH (μm):")
        print(f"  Max  length:   {cable_lengths.max():.3f}")
        print(f"  Mean length:   {cable_lengths.mean():.3f}")
        print(f"  Median length: {cable_lengths.median():.3f}")
        print(f"  Min  length:   {cable_lengths.min():.3f}")
        print(f"  Std  length:   {cable_lengths.std():.3f}")
    
    # E/I Distance Statistics
    if "min_geodesic_dist_to_e_syn" in syn_inh_cluster_df.columns:
        distances = syn_inh_cluster_df["min_geodesic_dist_to_e_syn"]
        print(f"\nMIN DISTANCE TO E-SYNAPSES (μm):")
        print(f"  Max  distance:   {distances.max():.3f}")
        print(f"  Mean distance:   {distances.mean():.3f}")
        print(f"  Median distance: {distances.median():.3f}")
        print(f"  Min  distance:   {distances.min():.3f}")
        print(f"  Std  distance:   {distances.std():.3f}")
    
    # E/I Relationship Statistics
    if "e_i_relationship" in syn_inh_cluster_df.columns:
        relationships = syn_inh_cluster_df["e_i_relationship"].value_counts()
        print(f"\nE/I RELATIONSHIPS:")
        for rel_type, count in relationships.items():
            print(f"  {rel_type}: {count} clusters")
    
    # Additional Summary Statistics
    print(f"\nSUMMARY:")
    print(f"  Total inhibitory clusters: {len(syn_inh_cluster_df)}")
    if "i_synapse_count" in syn_inh_cluster_df.columns:
        print(f"  Total inhibitory synapses: {syn_inh_cluster_df['i_synapse_count'].sum()}")
        if "minimal_cable_length" in syn_inh_cluster_df.columns:
            total_cable = syn_inh_cluster_df["minimal_cable_length"].sum()
            total_synapses = syn_inh_cluster_df["i_synapse_count"].sum()
            if total_cable > 0:
                print(f"  Total cable length: {total_cable:.2f} μm")
                print(f"  Overall density: {total_synapses / total_cable:.3f} synapses/μm")
    elif "Synapse_Count" in syn_inh_cluster_df.columns:
        print(f"  Total inhibitory synapses: {syn_inh_cluster_df['Synapse_Count'].sum()}")
        if "minimal_cable_length" in syn_inh_cluster_df.columns:
            total_cable = syn_inh_cluster_df["minimal_cable_length"].sum()
            total_synapses = syn_inh_cluster_df["Synapse_Count"].sum()
            if total_cable > 0:
                print(f"  Total cable length: {total_cable:.2f} μm")
                print(f"  Overall density: {total_synapses / total_cable:.3f} synapses/μm")
    
    print("="*60)
