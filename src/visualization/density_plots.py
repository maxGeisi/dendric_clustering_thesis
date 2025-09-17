# src/visualization/density_plots.py
"""
Density-based visualization functions.
"""
from __future__ import annotations
import plotly.graph_objects as go
import navis
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .visualization_config import VisualizationConfig, get_standard_colors, get_standard_markers


def plot_synapse_density(
    calculation_nodes: pd.DataFrame,
    neuron_skel,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create 3D visualization of synapse density with custom colorscale.
    
    Args:
        calculation_nodes: DataFrame with synapse density and coordinates
        neuron_skel: Navis neuron skeleton object
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    colors = get_standard_colors()
    
    # Define density range and custom colorscale
    density_min = float(calculation_nodes['synapse_density'].min())
    density_max = float(calculation_nodes['synapse_density'].max())
    
    custom_colorscale = [
        [0.0, colors["low_density"]],    # Black for low density
        [0.6, colors["medium_density"]], # Yellow for medium density
        [1.0, colors["high_density"]],   # Red for high density
    ]
    
    # Initialize 3D figure
    fig = go.Figure(layout=dict(
        width=config.figure_width,
        height=config.figure_height,
        title=f"Synapse Density - {config.neuron_id}",
        scene=dict(
            xaxis_title="X (μm)",
            yaxis_title="Y (μm)",
            zaxis_title="Z (μm)",
            aspectmode="data"
        )
    ))
    
    # Plot density nodes
    fig.add_trace(go.Scatter3d(
        x=calculation_nodes['x'],
        y=calculation_nodes['y'],
        z=calculation_nodes['z'],
        mode='markers',
        showlegend=False,
        marker=dict(
            size=5,  # Increased from 3 to 5 for better visibility
            color=calculation_nodes['synapse_density'],
            colorscale=custom_colorscale,
            cmin=density_min,
            cmax=density_max,
            colorbar=dict(
                title=dict(text="Density", side="right"),
                tickformat=".2f",
                ticks="outside",
                lenmode="fraction",
                len=0.6,
                x=1.02
            ),
            opacity=0.8
        )
    ))
    
    # Highlight peak node (maximum density)
    max_peak_idx = calculation_nodes['synapse_density'].idxmax()
    mp = calculation_nodes.loc[max_peak_idx]
    fig.add_trace(go.Scatter3d(
        x=[mp['x']], y=[mp['y']], z=[mp['z']],
        mode='markers',
        showlegend=False,
        marker=dict(
            size=6,
            color=colors["peak_node"],
            opacity=0.9
        )
    ))
    
    # Overlay skeleton (suppress legend)
    navis.plot3d(
        neuron_skel,
        fig=fig,
        color=colors["skeleton"],
        palette="viridis",
        legend=False,
        inline=False
    )
    
    # Keep only the synapse legend entries; hide any colorbars and neuron IDs
    keep = set()  # No legend entries to keep for density plots
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
        filename = config.get_filename("density", "synapse")
        output_path = config.get_output_path("density", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved density visualization: {output_path}")
    
    return fig


def plot_synapse_density_percentage_based(
    calculation_nodes: pd.DataFrame,
    neuron_skel,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create 3D visualization of synapse density with percentage-based coloring.
    
    Args:
        calculation_nodes: DataFrame with synapse density and coordinates
        neuron_skel: Navis neuron skeleton object
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    colors = get_standard_colors()
    
    # Calculate percentage-based density values
    density_values = calculation_nodes['synapse_density'].values
    density_min = np.min(density_values)
    density_max = np.max(density_values)
    
    # Convert to percentages of the range
    density_percentages = ((density_values - density_min) / (density_max - density_min)) * 100
    
    # Create percentage-based colorscale
    custom_colorscale = [
        [0.0, colors["low_density"]],    # Black for 0%
        [0.3, "#333333"],               # Dark grey for 30%
        [0.6, colors["medium_density"]], # Yellow for 60%
        [0.8, "#ff8000"],               # Orange for 80%
        [1.0, colors["high_density"]],   # Red for 100%
    ]
    
    # Initialize 3D figure
    fig = go.Figure(layout=dict(
        width=config.figure_width,
        height=config.figure_height,
        title=f"Synapse Density (Percentage-based) - {config.neuron_id}",
        scene=dict(
            xaxis_title="X (μm)",
            yaxis_title="Y (μm)",
            zaxis_title="Z (μm)",
            aspectmode="data"
        )
    ))
    
    # Plot density nodes with percentage-based coloring
    fig.add_trace(go.Scatter3d(
        x=calculation_nodes['x'],
        y=calculation_nodes['y'],
        z=calculation_nodes['z'],
        mode='markers',
        showlegend=False,
        marker=dict(
            size=5,  # Increased from 3 to 5 for better visibility
            color=density_percentages,
            colorscale=custom_colorscale,
            cmin=0,
            cmax=100,
            colorbar=dict(
                title=dict(text="Density (%)", side="right"),
                tickformat=".0f",
                ticks="outside",
                lenmode="fraction",
                len=0.6,
                x=1.02
            ),
            opacity=0.8
        )
    ))
    
    # Highlight peak node (maximum density)
    max_peak_idx = calculation_nodes['synapse_density'].idxmax()
    mp = calculation_nodes.loc[max_peak_idx]
    fig.add_trace(go.Scatter3d(
        x=[mp['x']], y=[mp['y']], z=[mp['z']],
        mode='markers',
        showlegend=False,
        marker=dict(
            size=6,
            color=colors["peak_node"],
            opacity=0.9
        )
    ))
    
    # Overlay skeleton (suppress legend)
    navis.plot3d(
        neuron_skel,
        fig=fig,
        color=colors["skeleton"],
        palette="viridis",
        legend=False,
        inline=False
    )
    
    # Keep only the synapse legend entries; hide any colorbars and neuron IDs
    keep = set()  # No legend entries to keep for density plots
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
        filename = config.get_filename("density", "percentage_based")
        output_path = config.get_output_path("density", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved percentage-based density visualization: {output_path}")
    
    return fig
