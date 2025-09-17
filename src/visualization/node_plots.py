# src/visualization/node_plots.py
"""
Node-based visualization functions.
"""
from __future__ import annotations
import plotly.graph_objects as go
import navis
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .visualization_config import VisualizationConfig, get_standard_colors, get_standard_markers


def plot_node_synapse_counts(
    node_counts: pd.Series,
    neuron_skel,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Visualize nodes with synapse counts, highlighting the node with maximum count.
    
    Args:
        node_counts: Series with synapse counts per node (aligned with neuron_skel.nodes.index)
        neuron_skel: Navis neuron skeleton object
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    colors = get_standard_colors()
    markers = get_standard_markers()
    
    # Reindex node_counts to match neuron skeleton nodes
    node_counts = node_counts.reindex(neuron_skel.nodes.index, fill_value=0)
    
    # Find nodes with synapses and maximum count
    nodes_above_zero = node_counts[node_counts > 0]
    max_count = node_counts.max()
    max_node = node_counts.idxmax()
    
    # Get coordinates for nodes with synapses
    nodes_to_plot = neuron_skel.nodes.loc[nodes_above_zero.index]
    max_node_coord = neuron_skel.nodes.loc[max_node]
    
    # Initialize 3D figure
    fig = go.Figure(layout=dict(
        width=config.figure_width,
        height=config.figure_height,
        title=f"Node Synapse Counts - {config.neuron_id}",
        scene=dict(
            xaxis_title="X (μm)",
            yaxis_title="Y (μm)",
            zaxis_title="Z (μm)",
            aspectmode="data"
        )
    ))
    
    # Plot nodes with synapses (blue dots)
    fig.add_trace(go.Scatter3d(
        x=nodes_to_plot['x'],
        y=nodes_to_plot['y'],
        z=nodes_to_plot['z'],
        mode='markers',
        marker=dict(
            size=markers["nodes_with_synapses"]["size"],
            color=markers["nodes_with_synapses"]["color"],
            opacity=0.8
        ),
        name='Nodes with >0 Synapses'
    ))
    
    # Highlight the node with maximum synapse count (red dot)
    fig.add_trace(go.Scatter3d(
        x=[max_node_coord['x']],
        y=[max_node_coord['y']],
        z=[max_node_coord['z']],
        mode='markers',
        marker=dict(
            size=markers["max_synapse_node"]["size"],
            color=markers["max_synapse_node"]["color"],
            opacity=0.9
        ),
        name=f'Highest synapse count: Node {max_count}'
    ))
    
    # Plot the dendritic tree skeleton
    navis.plot3d(
        neuron_skel,
        fig=fig,
        color=colors["skeleton_grey"],
        palette='viridis',
        legend=False,
        inline=False
    )
    
    # Keep only the synapse legend entries; hide any colorbars and neuron IDs
    keep = {"Nodes with >0 Synapses", "Highest synapse count: Node"}
    for tr in fig.data:
        # Hide neuron ID entries (they typically contain long alphanumeric strings)
        if tr.name and len(tr.name) > 20 and any(c.isdigit() for c in tr.name):
            tr.showlegend = False
        else:
            tr.showlegend = (tr.name in keep or (tr.name and "Highest synapse count" in tr.name))
        if hasattr(tr, "marker") and hasattr(tr.marker, "showscale"):
            tr.marker.showscale = False
    fig.update_layout(coloraxis_showscale=False, legend_title_text="")
    
    # Save plot if requested
    if save_plot:
        filename = config.get_filename("node_counts", "synapses")
        output_path = config.get_output_path("nodes", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved node counts visualization: {output_path}")
    
    return fig


def plot_node_synapse_counts_simple(
    node_counts: pd.Series,
    neuron_skel,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Simplified version showing only nodes with synapses (no max node highlight).
    
    Args:
        node_counts: Series with synapse counts per node
        neuron_skel: Navis neuron skeleton object
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    colors = get_standard_colors()
    markers = get_standard_markers()
    
    # Reindex node_counts to match neuron skeleton nodes
    node_counts = node_counts.reindex(neuron_skel.nodes.index, fill_value=0)
    
    # Find nodes with synapses
    nodes_above_zero = node_counts[node_counts > 0]
    nodes_to_plot = neuron_skel.nodes.loc[nodes_above_zero.index]
    
    # Initialize 3D figure
    fig = go.Figure(layout=dict(
        width=config.figure_width,
        height=config.figure_height,
        title=f"Nodes with Synapses - {config.neuron_id}",
        scene=dict(
            xaxis_title="X (μm)",
            yaxis_title="Y (μm)",
            zaxis_title="Z (μm)",
            aspectmode="data"
        )
    ))
    
    # Plot nodes with synapses (blue dots)
    fig.add_trace(go.Scatter3d(
        x=nodes_to_plot['x'],
        y=nodes_to_plot['y'],
        z=nodes_to_plot['z'],
        mode='markers',
        marker=dict(
            size=markers["nodes_with_synapses"]["size"],
            color=markers["nodes_with_synapses"]["color"],
            opacity=0.8
        ),
        name='Nodes with >0 Synapses'
    ))
    
    # Plot the dendritic tree skeleton
    navis.plot3d(
        neuron_skel,
        fig=fig,
        color=colors["skeleton_grey"],
        palette='viridis',
        legend=False,
        inline=False
    )
    
    # Keep only the synapse legend entries; hide any colorbars and neuron IDs
    keep = {"Nodes with >0 Synapses"}
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
        filename = config.get_filename("node_counts", "simple")
        output_path = config.get_output_path("nodes", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved simple node counts visualization: {output_path}")
    
    return fig
