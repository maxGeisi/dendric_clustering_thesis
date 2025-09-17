# src/visualization/advanced_histogram_plots.py
"""
Advanced histogram plotting functions for dendric clustering analysis.
Creates histograms with bar separation between clusters with/without I synapses,
distance distribution plots, and inhibitory cluster histograms.
"""
from __future__ import annotations
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from scipy.sparse.csgraph import minimum_spanning_tree

from .visualization_config import VisualizationConfig


def plot_excitatory_cluster_histogram_with_i_separation(
    cluster_df: pd.DataFrame,
    metric: str,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create histogram of excitatory clusters with bar separation between clusters with/without I synapses.
    Create histogram of excitatory clusters with bar separation between clusters with/without I synapses.
    
    Args:
        cluster_df: DataFrame with excitatory cluster information including has_I_associated
        metric: Metric to plot ('e_synapse_count', 'cluster_density', 'minimal_cable_length')
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    # Check if has_I_associated column exists, create it if missing
    if 'has_I_associated' not in cluster_df.columns:
        print(f"Warning: 'has_I_associated' column not found in cluster_df. Creating simple histogram.")
        return plot_simple_excitatory_histogram(cluster_df, metric, config, save_plot)
    
    # Create separate data for clusters with and without I synapses
    clusters_with_i = cluster_df[cluster_df['has_I_associated'] == True]
    clusters_without_i = cluster_df[cluster_df['has_I_associated'] == False]
    
    # Create the histogram with color separation
    fig = go.Figure()
    
    # Add histogram for clusters without I synapses (blue)
    if len(clusters_without_i) > 0:
        fig.add_trace(go.Histogram(
            x=clusters_without_i[metric],
            name='Clusters without I synapses',
            marker_color='lightblue',
            marker_line_color='darkblue',
            marker_line_width=1,
            opacity=0.7,
            nbinsx=40
        ))
    
    # Add histogram for clusters with I synapses (red)
    if len(clusters_with_i) > 0:
        fig.add_trace(go.Histogram(
            x=clusters_with_i[metric],
            name='Clusters with I synapses',
            marker_color='lightcoral',
            marker_line_color='darkred',
            marker_line_width=1,
            opacity=0.7,
            nbinsx=40
        ))
    
    # Set layout
    metric_labels = {
        'e_synapse_count': 'Number of Excitatory Synapses per Cluster',
        'cluster_density': 'Cluster Density (synapses/μm)',
        'minimal_cable_length': 'Minimal Cable Length (μm)'
    }
    
    metric_titles = {
        'e_synapse_count': 'Distribution of Excitatory Synapse Counts (After Filtering) - With/Without I Synapses',
        'cluster_density': 'Distribution of Excitatory Cluster Density (After Filtering) - With/Without I Synapses',
        'minimal_cable_length': 'Distribution of Excitatory Cluster Cable Length (After Filtering) - With/Without I Synapses'
    }
    
    fig.update_layout(
        title=metric_titles[metric],
        xaxis_title=metric_labels[metric],
        yaxis_title='Number of Clusters',
        barmode='overlay',
        bargap=0.1,
        margin=dict(l=60, r=20, t=100, b=40)
    )
    
    # Add mean and median lines for clusters WITH I synapses
    if len(clusters_with_i) > 0:
        mean_with_i = clusters_with_i[metric].mean()
        median_with_i = clusters_with_i[metric].median()
        
        # Add mean line for clusters with I (blue dotted)
        fig.add_vline(
            x=mean_with_i,
            line_dash="dot",
            line_color="blue",
            line_width=2
        )
        
        # Add median line for clusters with I (red dotted)
        fig.add_vline(
            x=median_with_i,
            line_dash="dot",
            line_color="red",
            line_width=2
        )
    
    # Add mean and median lines for clusters WITHOUT I synapses
    if len(clusters_without_i) > 0:
        mean_without_i = clusters_without_i[metric].mean()
        median_without_i = clusters_without_i[metric].median()
        
        # Add mean line for clusters without I (light blue dotted)
        fig.add_vline(
            x=mean_without_i,
            line_dash="dot",
            line_color="lightblue",
            line_width=2
        )
        
        # Add median line for clusters without I (light red dotted)
        fig.add_vline(
            x=median_without_i,
            line_dash="dot",
            line_color="lightcoral",
            line_width=2
        )
    
    # Add statistics annotations outside the plot area
    if len(clusters_with_i) > 0 and len(clusters_without_i) > 0:
        mean_with_i = clusters_with_i[metric].mean()
        median_with_i = clusters_with_i[metric].median()
        mean_without_i = clusters_without_i[metric].mean()
        median_without_i = clusters_without_i[metric].median()
        
        # Add statistics text below the title
        fig.add_annotation(
            x=0.5, y=1.12,
            xref="paper", yref="paper",
            text=f"<b>Statistics:</b> Mean (with I) = {mean_with_i:.2f} | Median (with I) = {median_with_i:.2f} | Mean (without I) = {mean_without_i:.2f} | Median (without I) = {median_without_i:.2f}",
            showarrow=False,
            font=dict(size=10),
            align="center"
        )
    
    # Add statistics annotations
    total_clusters = len(cluster_df)
    clusters_with_i_count = len(clusters_with_i)
    clusters_without_i_count = len(clusters_without_i)
    
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"Total clusters: {total_clusters}<br>With I synapses: {clusters_with_i_count}<br>Without I synapses: {clusters_without_i_count}",
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    # Print statistics
    print(f"\n{metric.upper()} STATISTICS:")
    print(f"  Total clusters: {total_clusters}")
    print(f"  Clusters with I synapses: {clusters_with_i_count} ({clusters_with_i_count/total_clusters*100:.1f}%)")
    print(f"  Clusters without I synapses: {clusters_without_i_count} ({clusters_without_i_count/total_clusters*100:.1f}%)")
    
    if len(clusters_with_i) > 0:
        print(f"  With I - Mean: {clusters_with_i[metric].mean():.2f}, Median: {clusters_with_i[metric].median():.2f}")
    if len(clusters_without_i) > 0:
        print(f"  Without I - Mean: {clusters_without_i[metric].mean():.2f}, Median: {clusters_without_i[metric].median():.2f}")
    
    # Save to file
    if save_plot:
        filename = config.get_filename(f"excitatory_{metric}_with_i_separation", "")
        output_path = config.get_output_path("histo_plots", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        # Also save as SVG
        svg_path = output_path.with_suffix('.svg')
        fig.write_image(str(svg_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved excitatory {metric} histogram with I separation to {output_path}")
    
    return fig


def plot_simple_excitatory_histogram(
    cluster_df: pd.DataFrame,
    metric: str,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create simple histogram of excitatory clusters (fallback when has_I_associated is not available).
    
    Args:
        cluster_df: DataFrame with excitatory cluster information
        metric: Metric to plot ('e_synapse_count', 'cluster_density', 'minimal_cable_length')
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    fig = px.histogram(
        cluster_df,
        x=metric,
        nbins=40,
        title=f"Distribution of Excitatory Cluster {metric.replace('_', ' ').title()} (After Filtering)",
        labels={metric: metric.replace('_', ' ').title()},
        text_auto=True
    )
    
    fig.update_layout(
        yaxis_title="Number of Clusters",
        margin=dict(l=60, r=20, t=100, b=40)
    )
    
    # Add dark borders to bars
    fig.update_traces(marker_line_color='black', marker_line_width=1)
    
    # Add mean and median lines
    mean_val = cluster_df[metric].mean()
    median_val = cluster_df[metric].median()
    
    # Add mean line (blue dotted)
    fig.add_vline(
        x=mean_val,
        line_dash="dot",
        line_color="blue",
        line_width=2
    )
    
    # Add median line (red dotted)
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color="red",
        line_width=2
    )
    
    # Add statistics text below the title
    fig.add_annotation(
        x=0.5, y=1.12,
        xref="paper", yref="paper",
        text=f"<b>Statistics:</b> Mean = {mean_val:.2f} | Median = {median_val:.2f}",
        showarrow=False,
        font=dict(size=10),
        align="center"
    )
    
    # Print statistics
    print(f"\n{metric.upper()} STATISTICS:")
    print(f"  Total clusters: {len(cluster_df)}")
    print(f"  Mean: {cluster_df[metric].mean():.2f}")
    print(f"  Median: {cluster_df[metric].median():.2f}")
    print(f"  Max: {cluster_df[metric].max():.2f}")
    print(f"  Min: {cluster_df[metric].min():.2f}")
    
    # Save to file
    if save_plot:
        filename = config.get_filename(f"excitatory_{metric}_simple", "")
        output_path = config.get_output_path("histo_plots", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        # Also save as SVG
        svg_path = output_path.with_suffix('.svg')
        fig.write_image(str(svg_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved simple excitatory {metric} histogram to {output_path}")
    
    return fig


def plot_inhibitory_distance_distribution_filtered(
    syn_inh_df_filtered: pd.DataFrame,
    dynamical_cutoff: float,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create histogram of filtered inhibitory synapse distances to excitatory synapses.
    This is the additional distance histogram from the user's code.
    
    Args:
        syn_inh_df_filtered: DataFrame with filtered inhibitory synapses
        dynamical_cutoff: Distance cutoff threshold
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    # Create histogram
    fig = px.histogram(
        syn_inh_df_filtered,
        x="min_dist_e_syn_in_clu",
        nbins=100,
        range_x=[0, 10],
        labels={"min_dist_e_syn_in_clu": "Distance to E synapse (μm)"},
        title="Distribution of Filtered Inhibitory Synapse Distances (After Filtering)",
        text_auto=True
    )
    
    # Add dark borders to bars
    fig.update_traces(marker_line_color='black', marker_line_width=1)
    
    # Tweak axes and rename Y
    fig.update_layout(
        bargap=0.05,
        xaxis=dict(
            tick0=0,
            dtick=0.5,
            title_text="Minimal distance of I synapse to E synapse (in cluster) (μm)"
        ),
        yaxis_title="Number of Synapses"
    )
    
    # Add mean and median lines
    mean_dist = syn_inh_df_filtered["min_dist_e_syn_in_clu"].mean()
    median_dist = syn_inh_df_filtered["min_dist_e_syn_in_clu"].median()
    
    # Add mean line (blue dotted)
    fig.add_vline(
        x=mean_dist,
        line_dash="dot",
        line_color="blue",
        line_width=2
    )
    
    # Add median line (red dotted)
    fig.add_vline(
        x=median_dist,
        line_dash="dot",
        line_color="red",
        line_width=2
    )
    
    # Add statistics text below the title
    fig.add_annotation(
        x=0.5, y=1.12,
        xref="paper", yref="paper",
        text=f"<b>Statistics:</b> Mean = {mean_dist:.3f} μm | Median = {median_dist:.3f} μm | Cutoff = {dynamical_cutoff:.3f} μm",
        showarrow=False,
        font=dict(size=10),
        align="center"
    )
    
    # Add cutoff line
    fig.add_vline(
        x=dynamical_cutoff,
        line_dash="dash",
        line_color="orange",
        line_width=2,
        annotation_text=f"Cutoff = {dynamical_cutoff:.3f} μm",
        annotation_position="top right"
    )
    
    # Print statistics
    print(f"\nFILTERED DISTANCE DISTRIBUTION STATISTICS:")
    print(f"  Total filtered synapses: {len(syn_inh_df_filtered)}")
    print(f"  Mean distance: {mean_dist:.3f} μm")
    print(f"  Median distance: {median_dist:.3f} μm")
    print(f"  Dynamic cutoff: {dynamical_cutoff:.3f} μm")
    
    # Save to file
    if save_plot:
        filename = config.get_filename("inhibitory_distance_distribution_filtered", "")
        output_path = config.get_output_path("histo_plots", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        # Also save as SVG
        svg_path = output_path.with_suffix('.svg')
        fig.write_image(str(svg_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved filtered inhibitory distance distribution to {output_path}")
    
    return fig


def compute_inhibitory_cluster_metrics(
    cluster_df_inh: pd.DataFrame,
    geodesic_mat_full: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute minimal cable length and cluster density for inhibitory clusters.
    
    Args:
        cluster_df_inh: DataFrame with inhibitory cluster information
        geodesic_mat_full: Geodesic distance matrix
        
    Returns:
        DataFrame with added metrics
    """
    cluster_df_inh = cluster_df_inh.copy()
    
    def minimal_cable_length_i(nodes_in_cluster, geodesic_mat):
        """Compute minimal cable length for inhibitory cluster nodes."""
        if len(nodes_in_cluster) < 2:
            return 0.0
        
        # Extract the submatrix for nodes
        node_subset = [n for n in nodes_in_cluster if n in geodesic_mat.index]
        if len(node_subset) < 2:
            return 0.0
        
        submat = geodesic_mat.loc[node_subset, node_subset]
        
        # Compute MST
        T = minimum_spanning_tree(submat)
        
        # Sum up the MST edge weights
        cable_len = T.sum()
        return cable_len
    
    def calc_cluster_density_i(synapse_count, cable_length):
        """Calculate cluster density for inhibitory clusters."""
        if cable_length == 0:
            return 0
        else:
            return synapse_count / cable_length
    
    # Compute minimal cable length
    cluster_cable_lengths_i = []
    for idx, row in cluster_df_inh.iterrows():
        nodes = row["Associated_Nodes"]
        cable_length = minimal_cable_length_i(nodes, geodesic_mat_full)
        cluster_cable_lengths_i.append(cable_length)
    
    cluster_df_inh["minimal_cable_length"] = cluster_cable_lengths_i
    
    # Compute cluster density
    synapse_count_col = "i_synapse_count" if "i_synapse_count" in cluster_df_inh.columns else "Synapse_Count"
    cluster_df_inh["cluster_density"] = cluster_df_inh.apply(
        lambda row: calc_cluster_density_i(row[synapse_count_col], row["minimal_cable_length"]), 
        axis=1
    )
    
    return cluster_df_inh


def plot_inhibitory_cluster_histogram(
    cluster_df_inh: pd.DataFrame,
    metric: str,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create histogram of inhibitory cluster metrics.
    
    Args:
        cluster_df_inh: DataFrame with inhibitory cluster information
        metric: Metric to plot ('i_synapse_count', 'cluster_density', 'minimal_cable_length')
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    # Determine the correct column name
    if metric == 'i_synapse_count' and 'i_synapse_count' not in cluster_df_inh.columns:
        if 'Synapse_Count' in cluster_df_inh.columns:
            metric = 'Synapse_Count'
        else:
            print(f"Warning: Neither 'i_synapse_count' nor 'Synapse_Count' found in cluster_df_inh")
            return None
    
    fig = px.histogram(
        cluster_df_inh,
        x=metric,
        nbins=50,
        title=f"Distribution of Inhibitory Cluster {metric.replace('_', ' ').title()} (After Filtering)",
        labels={metric: metric.replace('_', ' ').title()},
        text_auto=True
    )
    
    fig.update_layout(
        yaxis_title="Number of Clusters",
        margin=dict(l=60, r=20, t=100, b=40)
    )
    
    # Add dark borders to bars
    fig.update_traces(marker_line_color='black', marker_line_width=1)
    
    # Add mean and median lines
    mean_val = cluster_df_inh[metric].mean()
    median_val = cluster_df_inh[metric].median()
    
    # Add mean line (blue dotted)
    fig.add_vline(
        x=mean_val,
        line_dash="dot",
        line_color="blue",
        line_width=2
    )
    
    # Add median line (red dotted)
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color="red",
        line_width=2
    )
    
    # Add statistics text below the title
    fig.add_annotation(
        x=0.5, y=1.12,
        xref="paper", yref="paper",
        text=f"<b>Statistics:</b> Mean = {mean_val:.2f} | Median = {median_val:.2f}",
        showarrow=False,
        font=dict(size=10),
        align="center"
    )
    
    # Set appropriate tick spacing
    if metric in ['i_synapse_count', 'Synapse_Count']:
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(dtick=20)
    elif metric == 'cluster_density':
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(dtick=40)
    elif metric == 'minimal_cable_length':
        fig.update_xaxes(dtick=0.5)
        fig.update_yaxes(dtick=20)
    
    fig.update_traces(marker_line_color='black', marker_line_width=1)
    
    # Print statistics
    print(f"\nINHIBITORY {metric.upper()} STATISTICS:")
    print(f"  Total inhibitory clusters: {len(cluster_df_inh)}")
    print(f"  Max: {cluster_df_inh[metric].max():.2f}")
    print(f"  Mean: {cluster_df_inh[metric].mean():.2f}")
    print(f"  Median: {cluster_df_inh[metric].median():.2f}")
    print(f"  Min: {cluster_df_inh[metric].min():.2f}")
    
    # Save to file
    if save_plot:
        filename = config.get_filename(f"inhibitory_{metric}_distribution", "")
        output_path = config.get_output_path("histo_plots", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        # Also save as SVG
        svg_path = output_path.with_suffix('.svg')
        fig.write_image(str(svg_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved inhibitory {metric} histogram to {output_path}")
    
    return fig


def create_all_advanced_histograms(
    cluster_df: pd.DataFrame,
    cluster_df_inh: pd.DataFrame = None,
    syn_inh_df_filtered: pd.DataFrame = None,
    dynamical_cutoff: float = None,
    geodesic_mat_full: pd.DataFrame = None,
    
    config: VisualizationConfig = None,
    save_plots: bool = True
) -> Dict[str, go.Figure]:
    """
    Create all advanced histogram visualizations.
    
    Args:
        cluster_df: DataFrame with excitatory cluster information
        cluster_df_inh: DataFrame with inhibitory cluster information (optional)
        syn_inh_df_filtered: DataFrame with filtered inhibitory synapses (optional)
        dynamical_cutoff: Distance cutoff threshold (optional)
        geodesic_mat_full: Geodesic distance matrix (optional)
        syn_exec_df: DataFrame with excitatory synapse data (for cutoff calculation)
        neuron_skel: Neuron skeleton object (for cable length)
        config: Visualization configuration
        save_plots: Whether to save all plots to files
        
    Returns:
        Dictionary of created figures
    """
    print("Creating all advanced histogram visualizations...")
    
    # Use the provided dynamical_cutoff (it should be calculated correctly in the processing pipeline)
    if dynamical_cutoff is not None:
        print(f"Using provided dynamical cutoff: {dynamical_cutoff:.3f} μm")
    else:
        print("Warning: No dynamical cutoff provided. Using default value.")
        dynamical_cutoff = 2.5  # Default fallback
    
    figures = {}
    
    # Excitatory cluster histograms with I separation
    excitatory_metrics = ['e_synapse_count', 'cluster_density', 'minimal_cable_length']
    
    for metric in excitatory_metrics:
        if metric in cluster_df.columns:
            print(f"\nCreating excitatory {metric} histogram with I separation...")
            figures[f'excitatory_{metric}_with_i'] = plot_excitatory_cluster_histogram_with_i_separation(
                cluster_df, metric, config, save_plots
            )
        else:
            print(f"Warning: Column '{metric}' not found in cluster_df")
    
    # Additional filtered distance distribution (if data available)
    if (syn_inh_df_filtered is not None and dynamical_cutoff is not None):
        print(f"\nCreating filtered inhibitory distance distribution...")
        figures['inhibitory_distance_filtered'] = plot_inhibitory_distance_distribution_filtered(
            syn_inh_df_filtered, dynamical_cutoff, config, save_plots
        )
    else:
        print("Skipping filtered inhibitory distance distribution (missing data)")
    
    # Inhibitory cluster histograms (if data available)
    if cluster_df_inh is not None:
        # Compute metrics if not already present
        if 'minimal_cable_length' not in cluster_df_inh.columns and geodesic_mat_full is not None:
            print("Computing inhibitory cluster metrics...")
            cluster_df_inh = compute_inhibitory_cluster_metrics(cluster_df_inh, geodesic_mat_full)
        
        inhibitory_metrics = ['i_synapse_count', 'cluster_density', 'minimal_cable_length']
        
        for metric in inhibitory_metrics:
            if metric in cluster_df_inh.columns or (metric == 'i_synapse_count' and 'Synapse_Count' in cluster_df_inh.columns):
                print(f"\nCreating inhibitory {metric} histogram...")
                fig = plot_inhibitory_cluster_histogram(cluster_df_inh, metric, config, save_plots)
                if fig is not None:
                    figures[f'inhibitory_{metric}'] = fig
            else:
                print(f"Warning: Column '{metric}' not found in cluster_df_inh")
    else:
        print("Skipping inhibitory cluster histograms (missing cluster_df_inh)")
    
    print(f"\nCreated {len(figures)} advanced histogram visualizations successfully!")
    print(f"All histograms saved to: {config.get_output_path('histo_plots', '')}")
    
    return figures
