# src/processing/i_to_e_analysis.py
"""
Main orchestrator for inhibitory to excitatory synapse analysis.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from .distance_analysis import (
    find_closest_excitatory_synapses,
    map_inhibitory_to_excitatory_clusters_by_distance,
    split_mixed_inhibitory_clusters_by_distance,
    compute_distances_within_clusters,
    analyze_e_i_relationships,
    compute_distance_statistics,
    apply_distance_cutoff
)
from .analysis_io import create_analysis_directories
from .cluster_processing import build_cluster_dataframe, compute_cluster_metrics


def print_data_naming_conventions(
    syn_inh_df: pd.DataFrame,
    syn_inh_cluster_df: pd.DataFrame,
    syn_inh_cluster_df_filtered: pd.DataFrame,
    syn_inh_cluster_df_final: pd.DataFrame,
    distance_stats: Dict[str, Any],
    relationship_stats: Dict[str, Any],
    overall_density: float,
    dynamic_cutoff: float
) -> None:
    """
    Print concise data naming conventions and contents for all DataFrames.
    """
    print("=" * 60)
    print("DATA NAMING CONVENTIONS")
    print("=" * 60)
    
    print(f"\nDATAFRAME CONTENTS:")
    print(f"   syn_inh_df: ALL inhibitory synapses (original, unprocessed)")
    print(f"     • {len(syn_inh_df):,} inhibitory synapses")
    print(f"     • Synapse-level data with mapping information")
    
    print(f"\n   syn_inh_cluster_df: VALID inhibitory synapses with E-cluster mapping (calculated: closest E-synapse, E-cluster assignment)")
    print(f"     • {len(syn_inh_cluster_df):,} inhibitory synapses")
    print(f"     • Synapse-level data that successfully mapped to E-clusters")
    
    print(f"\n   syn_inh_cluster_df_filtered: FILTERED inhibitory synapses after dynamic cutoff (calculated: based on distance to closest E-synapse)")
    print(f"     • {len(syn_inh_cluster_df_filtered):,} inhibitory synapses")
    print(f"     • Synapse-level data within distance threshold")
    
    print(f"\n   syn_inh_cluster_df_final: FILTERED inhibitory clusters (cluster-level statistics) ")
    print(f"     • {len(syn_inh_cluster_df_final):,} inhibitory clusters")
    print(f"     • Cluster-level data with statistics (size, density, cable length)")
    print(f"     • Only clusters with synapses after filtering")
    
    print(f"\nFILTERING PIPELINE:")
    print(f"   1. Original: {len(syn_inh_df):,} inhibitory synapses")
    print(f"   2. E-mapping: {len(syn_inh_cluster_df):,} VALID inhibitory synapses")
    print(f"   3. Dynamic cutoff: {len(syn_inh_cluster_df_filtered):,} FILTERED inhibitory synapses")
    print(f"   4. Cluster rebuilding: {len(syn_inh_cluster_df_final):,} FILTERED inhibitory clusters")
    
    print(f"\nKEY DISTINCTIONS:")
    print(f"   • VALID inhibitory synapses: {len(syn_inh_cluster_df):,} synapses that mapped to E-clusters (calculated: closest E-synapse, E-cluster assignment)")
    print(f"   • FILTERED inhibitory synapses: {len(syn_inh_cluster_df_filtered):,} synapses after dynamic cutoff (calculated: distance filtering, E/I relationships)")
    print(f"   • FILTERED inhibitory clusters: {len(syn_inh_cluster_df_final):,} clusters from filtered synapses (calculated: cluster metrics, cable length, density)")
    
    print(f"\nDYNAMIC CUTOFF CALCULATION:")
    print(f"   • Overall density: {overall_density:.6f} synapses/μm")
    print(f"   • Dynamic cutoff: 1.0/overall_density = {dynamic_cutoff:.3f} μm")
    
    print(f"\nKEY DATAFRAME FOR CLUSTER ASSIGNMENTS:")
    print(f"   syn_inh_cluster_df_filtered: Contains the CORRECT E/I cluster assignments (calculated: distance filtering, E/I relationships)")
    print(f"     • Each row = one inhibitory synapse")
    print(f"     • cluster_id_exec = which excitatory cluster this I-synapse belongs to")
    print(f"     • cluster_id_inh = which inhibitory cluster this I-synapse belongs to")
    print(f"     • min_dist_e_syn_in_clu = distance to closest E-synapse in the assigned E-cluster")
    
    print("=" * 60)


def run_complete_i_to_e_analysis(
    syn_inh_df: pd.DataFrame,
    syn_exec_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    clusters_dict_inh: Dict[int, list],
    peak_to_cluster_id_inh: Dict[int, int],
    new_cluster_id_to_peak_inh: Dict[int, int],
    geodesic_mat_full: pd.DataFrame,
    neuron_skel,
    output_base_dir: Path,
    neuron_id: str,
    overall_density: float
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Run complete inhibitory to excitatory synapse analysis pipeline.
    
    Args:
        syn_inh_df: DataFrame with inhibitory synapse data
        syn_exec_df: DataFrame with excitatory synapse data
        cluster_df: DataFrame with excitatory cluster data
        clusters_dict_inh: Dictionary with inhibitory cluster information
        peak_to_cluster_id_inh: Mapping from peaks to inhibitory cluster IDs
        new_cluster_id_to_peak_inh: Mapping from new cluster IDs to peaks
        geodesic_mat_full: Geodesic distance matrix
        neuron_skel: Neuron skeleton object
        output_base_dir: Base directory for output files
        neuron_id: Neuron identifier
        overall_density: Overall synapse density (synapses/μm)
        
    Returns:
        Tuple of (filtered_synapses, filtered_clusters, distance_stats, relationship_stats)
    """
    # =============================================================================
    # STEP 1: Create output directories
    # =============================================================================
    analysis_dirs = create_analysis_directories(output_base_dir)
    
    # =============================================================================
    # STEP 2: Find closest excitatory synapses
    # =============================================================================
    syn_inh_df = find_closest_excitatory_synapses(syn_inh_df, syn_exec_df, geodesic_mat_full)
    
    # =============================================================================
    # STEP 3: Map inhibitory synapses to excitatory clusters
    # =============================================================================
    syn_inh_df = map_inhibitory_to_excitatory_clusters_by_distance(syn_inh_df, syn_exec_df, cluster_df)
    valid_mappings = (syn_inh_df['cluster_id_exec'] != -1).sum()
    
    # =============================================================================
    # STEP 4: Split mixed inhibitory clusters
    # =============================================================================
    valid_exec_ids = set(cluster_df["cluster_id"])
    syn_inh_df = split_mixed_inhibitory_clusters_by_distance(syn_inh_df, valid_exec_ids)
    
    # =============================================================================
    # STEP 5: Compute distances within clusters
    # =============================================================================
    syn_inh_df = compute_distances_within_clusters(syn_inh_df, syn_exec_df, geodesic_mat_full)
    
    # =============================================================================
    # STEP 6: Create filtered datasets (VALID inhibitory synapses)
    # =============================================================================
    syn_inh_cluster_df = syn_inh_df.dropna(subset=["min_dist_e_syn_in_clu"]).copy()
    
    # =============================================================================
    # STEP 7: Analyze E/I relationships
    # =============================================================================
    relationship_stats = analyze_e_i_relationships(syn_inh_cluster_df)
    
    # =============================================================================
    # STEP 8: Compute distance statistics
    # =============================================================================
    distance_stats = compute_distance_statistics(syn_inh_cluster_df)
    
    # =============================================================================
    # STEP 9: Apply DYNAMIC cutoff (matches original dc_initial_algo.py)
    # =============================================================================
    dynamic_cutoff = 1.0 / overall_density
    syn_inh_cluster_df_filtered = apply_distance_cutoff(syn_inh_cluster_df, dynamic_cutoff)
    
    # =============================================================================
    # STEP 10: Rebuild cluster DataFrames with filtered data
    # =============================================================================
    # Only include clusters that have synapses in the filtered dataset
    filtered_cluster_ids = set(syn_inh_cluster_df_filtered['cluster_id_inh'].unique())
    
    # Filter the clusters_dict to only include clusters with filtered synapses
    filtered_clusters_dict = {peak: nodes for peak, nodes in clusters_dict_inh.items() 
                             if peak_to_cluster_id_inh.get(peak) in filtered_cluster_ids}
    
    # Filter the mappings to only include relevant clusters
    filtered_peak_to_cluster_id = {peak: cid for peak, cid in peak_to_cluster_id_inh.items() 
                                  if cid in filtered_cluster_ids}
    filtered_cluster_id_to_peak = {cid: peak for peak, cid in filtered_peak_to_cluster_id.items()}
    
    syn_inh_cluster_df_final = build_cluster_dataframe(
        syn_inh_cluster_df_filtered, filtered_clusters_dict, filtered_peak_to_cluster_id, filtered_cluster_id_to_peak, 
        syn_id_col="id", cluster_id_col="cluster_id_inh", synapse_type="inhibitory"
    )
    
    syn_inh_cluster_df_final = compute_cluster_metrics(
        syn_inh_cluster_df_final, syn_inh_cluster_df_filtered, geodesic_mat_full, neuron_skel
    )
    
    # =============================================================================
    # STEP 11: Print data naming conventions and contents
    # =============================================================================
    print_data_naming_conventions(
        syn_inh_df, syn_inh_cluster_df, syn_inh_cluster_df_filtered, 
        syn_inh_cluster_df_final, distance_stats, relationship_stats,
        overall_density, dynamic_cutoff
    )
    
    return syn_inh_cluster_df_filtered, syn_inh_cluster_df_final, distance_stats, relationship_stats


