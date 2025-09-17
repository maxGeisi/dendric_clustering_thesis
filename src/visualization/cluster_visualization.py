# src/visualization/cluster_visualization.py
"""
Cluster-based visualization functions.
"""
from __future__ import annotations
import plotly.graph_objects as go
import navis
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .visualization_config import VisualizationConfig, get_standard_colors


def plot_clusters_by_synapse_count(
    syn_exec_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    neuron_skel,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create 3D visualization of excitatory clusters colored by synapse count (percentage-based).
    
    Args:
        syn_exec_df: DataFrame with excitatory synapses and cluster assignments
        cluster_df: DataFrame with cluster information including e_synapse_count
        neuron_skel: Navis neuron skeleton object
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    colors = get_standard_colors()
    
    # Calculate percentage-based thresholds for synapse counts
    synapse_counts = cluster_df['e_synapse_count'].values
    min_count = np.min(synapse_counts)
    max_count = np.max(synapse_counts)
    
    # Define percentage-based thresholds
    p25 = np.percentile(synapse_counts, 25)
    p50 = np.percentile(synapse_counts, 50)
    p75 = np.percentile(synapse_counts, 75)
    
    def get_cluster_color(count):
        """Get color based on percentage of synapse count range."""
        if count <= p25:
            return 'black'
        elif count <= p50:
            return 'turquoise'
        elif count <= p75:
            return 'yellow'
        else:
            return 'red'
    
    # Create legend entries with actual count ranges
    color_legend = [
        (f"≤{int(p25)} synapses", "black"),
        (f"{int(p25)+1}–{int(p50)} synapses", "turquoise"),
        (f"{int(p50)+1}–{int(p75)} synapses", "yellow"),
        (f">{int(p75)} synapses", "red"),
    ]
    
    # Initialize 3D figure
    fig = go.Figure(layout=dict(
        width=config.figure_width,
        height=config.figure_height,
        title=f"Excitatory Clusters by Synapse Count - {config.neuron_id}",
        scene=dict(
            xaxis_title="X (μm)",
            yaxis_title="Y (μm)",
            zaxis_title="Z (μm)",
            aspectmode="data"
        )
    ))
    
    # Add dummy legend entries
    for label, color in color_legend:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=4, color=color),
            name=label
        ))
    
    # Build lookup from cluster_id to synapse count
    count_map = dict(zip(cluster_df['e_cluster_id'], cluster_df['e_synapse_count']))
    
    # Plot each cluster
    for cid, group in syn_exec_df.groupby("cluster_id"):
        # Look up synapse count (default to 0 if not found)
        cluster_count = count_map.get(cid, 0)
        color = get_cluster_color(cluster_count)
        
        fig.add_trace(go.Scatter3d(
            x=group["Epos3DX"],
            y=group["Epos3DY"],
            z=group["Epos3DZ"],
            mode='markers',
            marker=dict(size=3, color=color, opacity=0.8),
            name=f"Cluster {cid} (n={cluster_count})",
            showlegend=False
        ))
    
    # Overlay neuron skeleton
    navis.plot3d(
        neuron_skel,
        fig=fig,
        color=colors["skeleton_grey"],
        palette='viridis',
        legend=False,
        inline=False
    )
    
    # Keep only the synapse legend entries; hide any colorbars and neuron IDs
    keep = set()  # No specific legend entries to keep for cluster plots
    for tr in fig.data:
        # Hide neuron ID entries (they typically contain long alphanumeric strings)
        if tr.name and len(tr.name) > 20 and any(c.isdigit() for c in tr.name):
            tr.showlegend = False
        else:
            tr.showlegend = (tr.name in keep)
        if hasattr(tr, "marker") and hasattr(tr.marker, "showscale"):
            tr.marker.showscale = False
    fig.update_layout(coloraxis_showscale=False, legend_title_text="")
    
    # Save plot if requested
    if save_plot:
        filename = config.get_filename("clusters", "synapse_count")
        output_path = config.get_output_path("clusters", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved cluster visualization: {output_path}")
    
    return fig


def plot_clusters_by_density(
    syn_exec_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    neuron_skel,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create 3D visualization of excitatory clusters colored by density (percentage-based).
    
    Args:
        syn_exec_df: DataFrame with excitatory synapses and cluster assignments
        cluster_df: DataFrame with cluster information including cluster_density
        neuron_skel: Navis neuron skeleton object
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    colors = get_standard_colors()
    
    # Calculate percentage-based thresholds for cluster density
    densities = cluster_df['cluster_density'].values
    min_density = np.min(densities)
    max_density = np.max(densities)
    
    # Define percentage-based thresholds
    p25 = np.percentile(densities, 25)
    p50 = np.percentile(densities, 50)
    p75 = np.percentile(densities, 75)
    
    def get_cluster_color(density):
        """Get color based on percentage of density range."""
        if density <= p25:
            return 'black'
        elif density <= p50:
            return 'turquoise'
        elif density <= p75:
            return 'yellow'
        else:
            return 'red'
    
    # Create legend entries with actual density ranges
    color_legend = [
        (f"≤{p25:.1f} density", "black"),
        (f"{p25:.1f}–{p50:.1f} density", "turquoise"),
        (f"{p50:.1f}–{p75:.1f} density", "yellow"),
        (f">{p75:.1f} density", "red"),
    ]
    
    # Initialize 3D figure
    fig = go.Figure(layout=dict(
        width=config.figure_width,
        height=config.figure_height,
        title=f"Excitatory Clusters by Density - {config.neuron_id}",
        scene=dict(
            xaxis_title="X (μm)",
            yaxis_title="Y (μm)",
            zaxis_title="Z (μm)",
            aspectmode="data"
        )
    ))
    
    # Add dummy legend entries
    for label, color in color_legend:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=4, color=color),
            name=label
        ))
    
    # Build lookup from cluster_id to density
    density_map = dict(zip(cluster_df['e_cluster_id'], cluster_df['cluster_density']))
    
    # Plot each cluster
    for cid, group in syn_exec_df.groupby("cluster_id"):
        # Look up density (default to 0 if not found)
        cluster_density = density_map.get(cid, 0)
        color = get_cluster_color(cluster_density)
        
        fig.add_trace(go.Scatter3d(
            x=group["Epos3DX"],
            y=group["Epos3DY"],
            z=group["Epos3DZ"],
            mode='markers',
            marker=dict(size=3, color=color, opacity=0.8),
            name=f"Cluster {cid} (d={cluster_density:.1f})",
            showlegend=False
        ))
    
    # Overlay neuron skeleton
    navis.plot3d(
        neuron_skel,
        fig=fig,
        color=colors["skeleton_grey"],
        palette='viridis',
        legend=False,
        inline=False
    )
    
    # Keep only the synapse legend entries; hide any colorbars and neuron IDs
    keep = set()  # No specific legend entries to keep for cluster plots
    for tr in fig.data:
        # Hide neuron ID entries (they typically contain long alphanumeric strings)
        if tr.name and len(tr.name) > 20 and any(c.isdigit() for c in tr.name):
            tr.showlegend = False
        else:
            tr.showlegend = (tr.name in keep)
        if hasattr(tr, "marker") and hasattr(tr.marker, "showscale"):
            tr.marker.showscale = False
    fig.update_layout(coloraxis_showscale=False, legend_title_text="")
    
    # Save plot if requested
    if save_plot:
        filename = config.get_filename("clusters", "density")
        output_path = config.get_output_path("clusters", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved cluster density visualization: {output_path}")
    
    return fig
