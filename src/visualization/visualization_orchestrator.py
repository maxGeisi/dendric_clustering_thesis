# src/visualization/visualization_orchestrator.py
"""
Orchestrator functions for creating comprehensive visualizations.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from .visualization_config import create_visualization_config
from .synapse_plots import plot_3d_synapses, plot_excitatory_synapses_only, plot_distance_to_soma
from .node_plots import plot_node_synapse_counts, plot_node_synapse_counts_simple
from .density_plots import plot_synapse_density, plot_synapse_density_percentage_based
from .cluster_visualization import plot_clusters_by_synapse_count, plot_clusters_by_density
from .histogram_plots import create_all_histograms, create_excitatory_histograms
from .advanced_histogram_plots import create_all_advanced_histograms
from .statistical_plots import create_all_statistical_plots
from .intra_inter_distance_plots import (
    plot_intra_vs_inter_cluster_distances,
    plot_intra_cluster_distance_histogram,
    plot_inter_cluster_distance_histogram
)
from .branch_plots import (
    plot_branch_length_distribution,
    plot_branch_points_3d,
    plot_branch_volume_correlation,
    plot_branch_synapse_count_correlation,
    plot_branch_filtering_summary,
    plot_branch_synapses_3d,
    plot_branch_synapses_by_cluster,
    plot_synapse_distance_to_soma_histogram,
    plot_branch_density_analysis,
    plot_branch_inhibitory_clusters,
    plot_branch_by_ecluster_enhanced
)


def create_all_visualizations(
    syn_exec_df: pd.DataFrame,
    syn_inh_df: pd.DataFrame,
    calculation_nodes: pd.DataFrame,
    node_counts: pd.Series,
    neuron_skel,
    neuron_id: str,
    output_base_dir: Path,
    cluster_df: pd.DataFrame = None,
    show_inhibitory: bool = True,
    save_plots: bool = True
) -> dict:
    """
    Create all available visualizations for the analysis.
    
    Args:
        syn_exec_df: DataFrame with excitatory synapse data
        syn_inh_df: DataFrame with inhibitory synapse data
        calculation_nodes: DataFrame with density calculation nodes
        node_counts: Series with synapse counts per node
        neuron_skel: Neuron skeleton object
        neuron_id: ID of the neuron being analyzed
        output_base_dir: Base directory for output files
        cluster_df: DataFrame with cluster information (optional)
        show_inhibitory: Whether to include inhibitory visualizations
        save_plots: Whether to save plots to files
        
    Returns:
        Dictionary of created figures
    """
    # Ensure output_base_dir is a Path object
    if isinstance(output_base_dir, str):
        output_base_dir = Path(output_base_dir)
    
    config = create_visualization_config(
        base_output_dir=output_base_dir,
        neuron_id=neuron_id
    )
    
    figures = {}
    
    # Synapse visualizations
    if show_inhibitory:
        figures['synapses_3d'] = plot_3d_synapses(
            syn_exec_df, syn_inh_df, neuron_skel, config, save_plots
        )
    else:
        figures['synapses_3d'] = plot_excitatory_synapses_only(
            syn_exec_df, neuron_skel, config, save_plots
        )
    
    # Node visualizations
    figures['node_counts'] = plot_node_synapse_counts(
        node_counts, neuron_skel, config, save_plots
    )
    
    figures['node_counts_simple'] = plot_node_synapse_counts_simple(
        node_counts, neuron_skel, config, save_plots
    )
    
    # Density visualizations
    figures['density'] = plot_synapse_density(
        calculation_nodes, neuron_skel, config, save_plots
    )
    
    figures['density_percentage'] = plot_synapse_density_percentage_based(
        calculation_nodes, neuron_skel, config, save_plots
    )
    
    # figures['distance_to_soma'] = plot_distance_to_soma(
    #     syn_exec_df, syn_inh_df, neuron_skel, config, save_plots
    # )
    
    # Cluster visualizations (if cluster data available)
    if cluster_df is not None:
        figures['clusters_by_count'] = plot_clusters_by_synapse_count(
            syn_exec_df, cluster_df, neuron_skel, config, save_plots
        )
        
        figures['clusters_by_density'] = plot_clusters_by_density(
            syn_exec_df, cluster_df, neuron_skel, config, save_plots
        )
    
    print(f"\nCreated {len(figures)} visualizations successfully!")
    
    return figures


def create_advanced_histogram_visualizations(
    cluster_df: pd.DataFrame,
    neuron_id: str,
    output_base_dir: Path,
    cluster_df_inh: pd.DataFrame = None,
    syn_inh_df_filtered: pd.DataFrame = None,
    dynamical_cutoff: float = None,
    geodesic_mat_full: pd.DataFrame = None,
    
    save_plots: bool = True
) -> dict:
    """
    Create advanced histogram visualizations with I/E separation and distance distributions.
    
    Args:
        cluster_df: DataFrame with excitatory cluster information
        neuron_id: ID of the neuron being analyzed
        output_base_dir: Base directory for output files
        cluster_df_inh: DataFrame with inhibitory cluster information (optional)
        syn_inh_df_filtered: DataFrame with filtered inhibitory synapses (optional)
        dynamical_cutoff: Distance cutoff threshold (optional)
        geodesic_mat_full: Geodesic distance matrix (optional)
        syn_exec_df: DataFrame with excitatory synapse data (for cutoff calculation)
        neuron_skel: Neuron skeleton object (for cable length)
        save_plots: Whether to save plots to files
        
    Returns:
        Dictionary of created figures
    """
    # Ensure output_base_dir is a Path object
    if isinstance(output_base_dir, str):
        output_base_dir = Path(output_base_dir)
    
    config = create_visualization_config(
        base_output_dir=output_base_dir,
        neuron_id=neuron_id
    )
    
    figures = create_all_advanced_histograms(
        cluster_df=cluster_df,
        cluster_df_inh=cluster_df_inh,
        syn_inh_df_filtered=syn_inh_df_filtered,
        dynamical_cutoff=dynamical_cutoff,
        geodesic_mat_full=geodesic_mat_full,
        config=config,
        save_plots=save_plots
    )
    
    print(f"\nCreated {len(figures)} advanced histogram visualizations successfully!")
    
    return figures


def create_histogram_visualizations(
    cluster_df: pd.DataFrame,
    cluster_df_inh: pd.DataFrame,
    calculation_nodes_inh: pd.DataFrame,
    all_synapses: pd.DataFrame,
    dynamical_cutoff: float,
    neuron_id: str,
    output_base_dir: Path,
    save_plots: bool = True
) -> dict:
    """
    Create histogram visualizations for the analysis.
    
    Args:
        cluster_df: DataFrame with excitatory cluster information
        cluster_df_inh: DataFrame with inhibitory cluster information
        calculation_nodes_inh: DataFrame with inhibitory density nodes
        all_synapses: Combined DataFrame of inhibitory synapses
        dynamical_cutoff: Distance cutoff threshold
        neuron_id: ID of the neuron being analyzed
        output_base_dir: Base directory for output files
        save_plots: Whether to save plots to files
        
    Returns:
        Dictionary of created figures
    """
    # Ensure output_base_dir is a Path object
    if isinstance(output_base_dir, str):
        output_base_dir = Path(output_base_dir)
    
    config = create_visualization_config(
        base_output_dir=output_base_dir,
        neuron_id=neuron_id
    )
    
    figures = create_all_histograms(
        cluster_df=cluster_df,
        cluster_df_inh=cluster_df_inh,
        calculation_nodes_inh=calculation_nodes_inh,
        all_synapses=all_synapses,
        dynamical_cutoff=dynamical_cutoff,
        config=config,
        save_plots=save_plots
    )
    
    print(f"\nCreated {len(figures)} histogram visualizations successfully!")
    
    return figures


def create_excitatory_histogram_visualizations(
    cluster_df: pd.DataFrame,
    neuron_id: str,
    output_base_dir: Path,
    save_plots: bool = True
) -> dict:
    """
    Create excitatory-only histogram visualizations.
    
    Args:
        cluster_df: DataFrame with excitatory cluster information
        neuron_id: ID of the neuron being analyzed
        output_base_dir: Base directory for output files
        save_plots: Whether to save plots to files
        
    Returns:
        Dictionary of created figures
    """
    # Ensure output_base_dir is a Path object
    if isinstance(output_base_dir, str):
        output_base_dir = Path(output_base_dir)
    
    config = create_visualization_config(
        base_output_dir=output_base_dir,
        neuron_id=neuron_id
    )
    
    figures = create_excitatory_histograms(
        cluster_df=cluster_df,
        config=config,
        save_plots=save_plots
    )
    
    print(f"\nCreated {len(figures)} excitatory histogram visualizations successfully!")
    
    return figures


def create_statistical_comparison_visualizations(
    cluster_df: pd.DataFrame,
    neuron_id: str,
    output_base_dir: Path,
    save_plots: bool = True
) -> dict:
    """
    Create statistical comparison visualizations for cluster properties.
    
    Args:
        cluster_df: DataFrame with cluster information including has_I_associated column
        neuron_id: ID of the neuron being analyzed
        output_base_dir: Base directory for output files
        save_plots: Whether to save plots to files
        
    Returns:
        Dictionary of created figures
    """
    # Ensure output_base_dir is a Path object
    if isinstance(output_base_dir, str):
        output_base_dir = Path(output_base_dir)
    
    config = create_visualization_config(
        base_output_dir=output_base_dir,
        neuron_id=neuron_id
    )
    
    figures = create_all_statistical_plots(
        cluster_df=cluster_df,
        config=config,
        save_plots=save_plots
    )
    
    print(f"\nCreated {len(figures)} statistical comparison visualizations successfully!")
    
    return figures


def create_intra_inter_distance_visualizations(
    cluster_df: pd.DataFrame,
    neuron_id: str,
    output_base_dir: Path,
    save_plots: bool = True
) -> dict:
    """
    Create visualizations for intra-cluster vs inter-cluster distance analysis.
    
    Args:
        cluster_df: DataFrame with cluster information including intra and inter distances
        neuron_id: Neuron identifier
        output_base_dir: Base output directory
        save_plots: Whether to save plots
        
    Returns:
        Dictionary with created visualizations
    """
    print("=" * 80)
    print("CREATING INTRA-INTER CLUSTER DISTANCE VISUALIZATIONS")
    print("=" * 80)
    
    # Ensure output_base_dir is a Path object
    if isinstance(output_base_dir, str):
        output_base_dir = Path(output_base_dir)
    
    # Create visualization config
    config = create_visualization_config(
        base_output_dir=output_base_dir,
        neuron_id=neuron_id
    )
    
    results = {}
    
    try:
        # Check if required columns exist (check for both possible column name patterns)
        intra_col = 'median_intra_dist' if 'median_intra_dist' in cluster_df.columns else 'median_intra_dist_intra'
        inter_col = 'inter_median_dist' if 'inter_median_dist' in cluster_df.columns else 'inter_median_dist_inter'
        
        required_cols = [intra_col, inter_col]
        missing_cols = [col for col in required_cols if col not in cluster_df.columns]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print("   Skipping intra-inter distance visualizations")
            return results
        
        print(f"Available data:")
        print(f"   • Total clusters: {len(cluster_df)}")
        print(f"   • Clusters with intra distances: {len(cluster_df[cluster_df[intra_col] > 0])}")
        print(f"   • Clusters with inter distances: {len(cluster_df.dropna(subset=[inter_col]))}")
        
        # Debug: print column names to help troubleshoot
        print(f"   • Available columns: {list(cluster_df.columns)}")
        
        # 1. Intra vs Inter cluster distance comparison
        print(f"\nCreating intra vs inter cluster distance comparison...")
        fig1 = plot_intra_vs_inter_cluster_distances(cluster_df, config, save_plot=save_plots)
        results['intra_vs_inter_comparison'] = fig1
        
        # 2. Intra-cluster distance histogram
        print(f"\nCreating intra-cluster distance histogram...")
        fig2 = plot_intra_cluster_distance_histogram(cluster_df, config, save_plot=save_plots)
        results['intra_distance_histogram'] = fig2
        
        # 3. Inter-cluster distance histogram
        print(f"\nCreating inter-cluster distance histogram...")
        fig3 = plot_inter_cluster_distance_histogram(cluster_df, config, save_plot=save_plots)
        results['inter_distance_histogram'] = fig3
        
        print(f"\nCreated {len(results)} intra-inter distance visualizations successfully!")
        
    except Exception as e:
        print(f"Error creating intra-inter distance visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def create_branch_visualizations(
    neuron_splits: list,
    branch_df: pd.DataFrame,
    filtered_branch_df: pd.DataFrame,
    filter_stats: Dict[str, Any],
    syn_exec_df: pd.DataFrame,
    syn_inh_df: pd.DataFrame,
    neuron_skel,
    junctions_list: list,
    calculation_nodes: pd.DataFrame,
    neuron_id: str,
    output_base_dir: Path,
    save_plots: bool = True
) -> dict:
    """
    Create all branch analysis visualizations.
    
    Args:
        neuron_splits: List of branch skeleton objects
        branch_df: DataFrame with branch information
        filtered_branch_df: DataFrame with filtered branch information
        filter_stats: Dictionary with filtering statistics
        syn_exec_df: DataFrame with excitatory synapse data
        syn_inh_df: DataFrame with inhibitory synapse data
        neuron_skel: Neuron skeleton object
        junctions_list: List of junction node IDs
        calculation_nodes: DataFrame with node coordinates
        neuron_id: ID of the neuron being analyzed
        output_base_dir: Base directory for output files
        save_plots: Whether to save plots to files
        
    Returns:
        Dictionary with created figures
    """
    print(f"\nCreating branch analysis visualizations...")
    
    # Ensure output_base_dir is a Path object
    if isinstance(output_base_dir, str):
        output_base_dir = Path(output_base_dir)
    
    # Create visualization config
    config = create_visualization_config(output_base_dir, neuron_id)
    
    # Create branches directory
    branches_dir = config.base_output_dir / "branches"
    branches_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    try:
        # 1. Branch length distribution
        print(f"Creating branch length distribution plot...")
        fig1 = plot_branch_length_distribution(neuron_splits, config, save_plot=save_plots)
        results['branch_length_distribution'] = fig1
        
        # 2. Branch points 3D visualization
        print(f"Creating branch points 3D visualization...")
        fig2 = plot_branch_points_3d(
            neuron_skel, junctions_list, calculation_nodes, config, save_plot=save_plots
        )
        results['branch_points_3d'] = fig2
        
        # 3. Branch volume correlation
        print(f"Creating branch volume correlation plot...")
        fig3 = plot_branch_volume_correlation(
            filtered_branch_df, syn_exec_df, syn_inh_df, config, save_plot=save_plots
        )
        results['branch_volume_correlation'] = fig3
        
        # 4. Branch synapse count correlation
        print(f"Creating branch synapse count correlation plot...")
        fig4 = plot_branch_synapse_count_correlation(filtered_branch_df, config, save_plot=save_plots)
        results['branch_synapse_count_correlation'] = fig4
        
        # 5. Branch filtering summary
        print(f"Creating branch filtering summary plot...")
        fig5 = plot_branch_filtering_summary(filter_stats, config, save_plot=save_plots)
        results['branch_filtering_summary'] = fig5
        
        print(f"\nCreated {len(results)} branch analysis visualizations successfully!")
        
    except Exception as e:
        print(f"Error creating branch visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def create_advanced_branch_visualizations(
    branch_idx: int,
    neuron_splits: list,
    syn_exec_df: pd.DataFrame,
    syn_inh_df: pd.DataFrame,
    syn_inh_df_filtered: pd.DataFrame,
    cluster_df: pd.DataFrame,
    calculation_nodes: pd.DataFrame,
    geodesic_mat_full: pd.DataFrame,
    neuron_skel,
    cluster_df_inh: pd.DataFrame = None,
    neuron_id: str = "n3",
    output_base_dir: Path = None,
    save_plots: bool = True
) -> dict:
    """
    Create advanced branch visualizations for a specific branch.
    
    Args:
        branch_idx: Index of the branch to visualize
        neuron_splits: List of branch skeleton objects
        syn_exec_df: DataFrame with excitatory synapse data
        syn_inh_df: DataFrame with inhibitory synapse data
        cluster_df: DataFrame with cluster information
        calculation_nodes: DataFrame with density calculation nodes
        geodesic_mat_full: Full geodesic distance matrix
        neuron_skel: Neuron skeleton object
        cluster_df_inh: DataFrame with inhibitory cluster information (optional)
        neuron_id: ID of the neuron being analyzed
        output_base_dir: Base directory for output files
        save_plots: Whether to save plots to files
        
    Returns:
        Dictionary with created figures
    """
    print(f"\nCreating advanced branch visualizations for branch {branch_idx}...")
    
    # Ensure output_base_dir is a Path object
    if isinstance(output_base_dir, str):
        output_base_dir = Path(output_base_dir)
    
    # Create visualization config
    config = create_visualization_config(output_base_dir, neuron_id)
    
    # Create branches directory
    branches_dir = config.base_output_dir / "branches"
    branches_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    try:
        # 1. Branch synapses 3D visualization
        fig1 = plot_branch_synapses_3d(
            branch_idx, neuron_splits, syn_exec_df, syn_inh_df, config, save_plot=save_plots
        )
        results['branch_synapses_3d'] = fig1
        
        # 2. Branch synapses by cluster
        fig2 = plot_branch_synapses_by_cluster(
            branch_idx, neuron_splits, syn_exec_df, cluster_df, config, save_plot=save_plots
        )
        results['branch_synapses_by_cluster'] = fig2
        
        # 3. Synapse distance to soma histogram
        fig3 = plot_synapse_distance_to_soma_histogram(
            syn_exec_df, syn_inh_df, geodesic_mat_full, neuron_skel, config, save_plot=save_plots
        )
        results['synapse_distance_to_soma_histogram'] = fig3
        
        # 4. Branch density analysis
        fig4 = plot_branch_density_analysis(
            branch_idx, neuron_splits, syn_exec_df, calculation_nodes, geodesic_mat_full, config, save_plot=save_plots
        )
        results['branch_density_analysis'] = fig4

        fig5 = plot_branch_by_ecluster_enhanced(
            branch_idx, neuron_splits, syn_exec_df, syn_inh_df_filtered, cluster_df, calculation_nodes, geodesic_mat_full, config, margin=3.0, save_plot=save_plots
        )
        results['branch_e_clusters_gradient'] = fig5

        # 5. Branch inhibitory clusters (if inhibitory data available)
        if cluster_df_inh is not None:
            fig6 = plot_branch_inhibitory_clusters(
                branch_idx, neuron_splits, syn_exec_df, syn_inh_df, cluster_df_inh, neuron_skel, config, save_plot=save_plots
            )
            results['branch_inhibitory_clusters'] = fig6
        
        print(f"\nCreated {len(results)} advanced branch visualizations successfully!")
        
    except Exception as e:
        print(f"Error creating advanced branch visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    return results


