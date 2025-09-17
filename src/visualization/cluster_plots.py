# src/visualization/cluster_plots.py
"""
3D cluster visualization functions for dendric clustering analysis.
"""
from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
import navis
import pandas as pd
from typing import Dict, List, Tuple, Any
from pathlib import Path

def get_cluster_color_scheme(cluster_df: pd.DataFrame) -> Tuple[List[Tuple[str, str]], callable]:
    """
    Define percentage-based color scheme for cluster visualization.
    
    Args:
        cluster_df: DataFrame with cluster information to calculate percentiles
        
    Returns:
        Tuple of (color_legend, color_function)
    """
    import numpy as np
    
    # Calculate percentage-based thresholds for synapse counts
    synapse_counts = cluster_df['e_synapse_count'].values if 'e_synapse_count' in cluster_df.columns else cluster_df['i_synapse_count'].values
    p25 = np.percentile(synapse_counts, 25)
    p50 = np.percentile(synapse_counts, 50)
    p75 = np.percentile(synapse_counts, 75)
    
    color_legend = [
        (f"≤{int(p25)} synapses", "black"),
        (f"{int(p25)+1}–{int(p50)} synapses", "turquoise"),
        (f"{int(p50)+1}–{int(p75)} synapses", "yellow"),
        (f">{int(p75)} synapses", "red"),
    ]
    
    def get_cluster_color(count: int) -> str:
        """Get color based on percentage of synapse count range."""
        if count <= p25:
            return 'black'
        elif count <= p50:
            return 'turquoise'
        elif count <= p75:
            return 'yellow'
        else:
            return 'red'
    
    return color_legend, get_cluster_color

def plot_clusters_3d(
    neuron_skel,
    syn_exec_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    width: int = 1200,
    height: int = 800,
    show_legend: bool = True
) -> go.Figure:
    """
    Create 3D visualization of excitatory clusters.
    
    Args:
        neuron_skel: Navis neuron skeleton object
        syn_exec_df: DataFrame with excitatory synapse data
        cluster_df: DataFrame with cluster information
        width: Figure width
        height: Figure height
        show_legend: Whether to show legend
        
    Returns:
        Plotly figure object
    """
    color_legend, get_cluster_color = get_cluster_color_scheme(cluster_df)
    
    fig = go.Figure(layout=dict(width=width, height=height))
    
    # Add dummy legend entries
    if show_legend:
        for label, color in color_legend:
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=4, color=color),
                name=label
            ))
    
    # Create count mapping for clusters
    syn_count_col = "e_synapse_count" if "e_synapse_count" in cluster_df.columns else "i_synapse_count"
    count_map = cluster_df.set_index('cluster_id')[syn_count_col].to_dict()
    
    # Plot each cluster
    for cid, group in syn_exec_df.groupby("cluster_id"):
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
    
    # Add neuron skeleton
    navis.plot3d(neuron_skel, fig=fig, color="grey", palette='viridis', legend=False, inline=False)
    
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
    fig.update_layout(coloraxis_showscale=False, legend_title_text="", showlegend=show_legend)
    
    return fig

def plot_clusters_by_size(
    syn_exec_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    size_bins: List[Tuple[int, int, str]] = None
) -> go.Figure:
    """
    Plot clusters grouped by size categories.
    
    Args:
        syn_exec_df: DataFrame with excitatory synapse data
        cluster_df: DataFrame with cluster information
        size_bins: List of (min, max, color) tuples for size categories
        
    Returns:
        Plotly figure object
    """
    if size_bins is None:
        # Use percentage-based thresholds instead of hardcoded values
        import numpy as np
        synapse_counts = cluster_df[syn_count_col].values
        p25 = int(np.percentile(synapse_counts, 25))
        p50 = int(np.percentile(synapse_counts, 50))
        p75 = int(np.percentile(synapse_counts, 75))
        
        size_bins = [
            (1, p25, "black"),
            (p25+1, p50, "turquoise"),
            (p50+1, p75, "yellow"),
            (p75+1, 100, "red")
        ]
    
    # Determine synapse count column name
    syn_count_col = "e_synapse_count" if "e_synapse_count" in cluster_df.columns else "i_synapse_count"
    
    fig = go.Figure()
    
    for min_size, max_size, color in size_bins:
        # Find clusters in this size range
        clusters_in_range = cluster_df[
            (cluster_df[syn_count_col] >= min_size) & 
            (cluster_df[syn_count_col] <= max_size)
        ]['cluster_id'].tolist()
        
        if clusters_in_range:
            # Get synapses for these clusters
            syns_in_range = syn_exec_df[syn_exec_df['cluster_id'].isin(clusters_in_range)]
            
            fig.add_trace(go.Scatter3d(
                x=syns_in_range["Epos3DX"],
                y=syns_in_range["Epos3DY"],
                z=syns_in_range["Epos3DZ"],
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.8),
                name=f"{min_size}-{max_size} synapses (n={len(clusters_in_range)})"
            ))
    
    return fig

def plot_e_i_overlay(
    neuron_skel,
    syn_exec_df: pd.DataFrame,
    syn_inh_df: pd.DataFrame,
    width: int = 1200,
    height: int = 800
) -> go.Figure:
    """
    Plot excitatory and inhibitory synapses overlaid on neuron skeleton.
    
    Args:
        neuron_skel: Navis neuron skeleton object
        syn_exec_df: DataFrame with excitatory synapse data
        syn_inh_df: DataFrame with inhibitory synapse data
        width: Figure width
        height: Figure height
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(layout=dict(width=width, height=height))
    
    # Plot excitatory synapses
    fig.add_trace(go.Scatter3d(
        x=syn_exec_df["Epos3DX"],
        y=syn_exec_df["Epos3DY"],
        z=syn_exec_df["Epos3DZ"],
        mode='markers',
        marker=dict(size=1.5, color='blue'),
        name='Excitatory synapses'
    ))
    
    # Plot inhibitory synapses
    fig.add_trace(go.Scatter3d(
        x=syn_inh_df["Ipos3DX"],
        y=syn_inh_df["Ipos3DY"],
        z=syn_inh_df["Ipos3DZ"],
        mode='markers',
        marker=dict(size=2, color='red'),
        name='Inhibitory synapses'
    ))
    
    # Add neuron skeleton
    navis.plot3d(neuron_skel, fig=fig, color="green", palette="viridis", legend=False, inline=False)
    
    # Keep only the synapse legend entries; hide any colorbars and neuron IDs
    keep = {"Excitatory synapses", "Inhibitory synapses"}
    for tr in fig.data:
        # Hide neuron ID entries (they typically contain long alphanumeric strings)
        if tr.name and len(tr.name) > 20 and any(c.isdigit() for c in tr.name):
            tr.showlegend = False
        else:
            tr.showlegend = (tr.name in keep)
        if hasattr(tr, "marker") and hasattr(tr.marker, "showscale"):
            tr.marker.showscale = False
    fig.update_layout(coloraxis_showscale=False, legend_title_text="")
    
    return fig

def save_cluster_plot(fig: go.Figure, output_dir: Path, filename: str) -> Path:
    """
    Save cluster plot to file.
    
    Args:
        fig: Plotly figure object
        output_dir: Output directory path
        filename: Filename for the plot
        
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    # Save as PNG
    fig.write_image(str(output_path))
    
    return output_path
