# src/visualization/histogram_plots.py
"""
Comprehensive histogram plotting functions for dendric clustering analysis.
All histograms are saved to the histo_plots directory with proper naming conventions.
"""
from __future__ import annotations
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .visualization_config import VisualizationConfig


def plot_excitatory_synapse_count_histogram(
    cluster_df: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create histogram of excitatory cluster synapse counts with mean/median annotations.
    
    Args:
        cluster_df: DataFrame with excitatory cluster information
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    # Build the basic histogram
    fig_hist = px.histogram(
        cluster_df,
        x="e_synapse_count",
        nbins=40,
        title="Histogram of Excitatory Cluster Synapse Counts (After Clustering and Filtering)",
        labels={"e_synapse_count": "Number of Synapses per Cluster"},
        text_auto=True
    )
    
    # Compute mean & median
    mean_syn = cluster_df["e_synapse_count"].mean()
    median_syn = cluster_df["e_synapse_count"].median()

    # Add vertical dotted lines for mean (red) and median (blue)
    fig_hist.add_vline(x=mean_syn, line_dash="dot", line_color="red")
    fig_hist.add_vline(x=median_syn, line_dash="dot", line_color="blue")

    # Annotate mean just above the plot, slightly to the right
    fig_hist.add_annotation(
        x=mean_syn, y=1.02,
        xref="x", yref="paper",
        text=f"Mean = {mean_syn:.1f}",
        showarrow=False,
        font=dict(color="red"),
        xanchor="left", yanchor="bottom"
    )

    # Annotate median at the top edge, slightly to the left
    fig_hist.add_annotation(
        x=median_syn, y=1.00,
        xref="x", yref="paper",
        text=f"Median = {median_syn:.1f}",
        showarrow=False,
        font=dict(color="blue"),
        xanchor="right", yanchor="bottom"
    )

    # Tweak layout and tick spacing
    fig_hist.update_layout(
        yaxis_title="Number of Clusters",
        xaxis_title="Number of Synapses per Cluster",
        margin=dict(l=60, r=20, t=60, b=40)
    )
    fig_hist.update_xaxes(dtick=1)
    fig_hist.update_yaxes(dtick=10)
    fig_hist.update_traces(marker_line_color="black", marker_line_width=1)

    # Print summary stats
    print("Max  # syn:   ", cluster_df["e_synapse_count"].max())
    print("Mean # syn:  ", mean_syn)
    print("Median # syn:", median_syn)

    # Save to file
    if save_plot:
        filename = config.get_filename("cluster_synapse_count_distribution", "")
        output_path = config.get_output_path("histo_plots", filename)
        fig_hist.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        # Also save as SVG
        svg_path = output_path.with_suffix('.svg')
        fig_hist.write_image(str(svg_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved excitatory cluster synapse count histogram to {output_path}")

    return fig_hist


def plot_excitatory_density_histogram(
    cluster_df: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create histogram of excitatory cluster density distribution.
    
    Args:
        cluster_df: DataFrame with excitatory cluster information
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    # Plot histogram of cluster_density
    fig_hist = px.histogram(
        cluster_df,
        x="cluster_density",
        nbins=50,
        title="Distribution of Excitatory Cluster Density (After Clustering and Filtering)",
        labels={"cluster_density": "Cluster Density"},
        text_auto=True
    )

    # Compute mean & median
    mean_den = cluster_df["cluster_density"].mean()
    median_den = cluster_df["cluster_density"].median()

    # Add vertical dotted lines
    fig_hist.add_vline(x=mean_den, line_dash="dot", line_color="red")
    fig_hist.add_vline(x=median_den, line_dash="dot", line_color="blue")

    # Annotate mean just above the plot
    fig_hist.add_annotation(
        x=mean_den, y=1.03,
        xref="x", yref="paper",
        text=f"Mean = {mean_den:.2f}",
        showarrow=False,
        font=dict(color="red"),
        xanchor="left", yanchor="bottom"
    )

    # Annotate median at the top edge
    fig_hist.add_annotation(
        x=median_den, y=1.00,
        xref="x", yref="paper",
        text=f"Median = {median_den:.2f}",
        showarrow=False,
        font=dict(color="blue"),
        xanchor="right", yanchor="bottom"
    )

    # Tweak axes & styling
    fig_hist.update_layout(
        yaxis_title="Number of Clusters",
        xaxis_title="Cluster Density",
        margin=dict(l=60, r=20, t=60, b=40)
    )
    fig_hist.update_xaxes(dtick=1)
    fig_hist.update_yaxes(dtick=20)
    fig_hist.update_traces(marker_line_color="black", marker_line_width=1)

    # Print stats
    print("Max density:   ", cluster_df["cluster_density"].max())
    print("Mean density:  ", mean_den)
    print("Median density:", median_den)

    # Save to file
    if save_plot:
        filename = config.get_filename("cluster_density_distribution", "")
        output_path = config.get_output_path("histo_plots", filename)
        fig_hist.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        # Also save as SVG
        svg_path = output_path.with_suffix('.svg')
        fig_hist.write_image(str(svg_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved excitatory cluster density histogram to {output_path}")

    return fig_hist


def plot_excitatory_cable_length_histogram(
    cluster_df: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create histogram of excitatory cluster cable length distribution.
    
    Args:
        cluster_df: DataFrame with excitatory cluster information
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    # Plot histogram of cluster sizes (minimal_cable_length)
    fig_hist = px.histogram(
        cluster_df,
        x="minimal_cable_length",
        nbins=50,
        title="Distribution of Excitatory Cluster Cable Length (After Clustering and Filtering)",
        labels={"minimal_cable_length": "Cluster Cable Length (μm)"},
        text_auto=True
    )
    
    # Compute mean & median
    mean_size = cluster_df["minimal_cable_length"].mean()
    median_size = cluster_df["minimal_cable_length"].median()

    # Add vertical lines
    fig_hist.add_vline(x=mean_size, line_dash="dot", line_color="red")
    fig_hist.add_vline(x=median_size, line_dash="dot", line_color="blue")

    # Annotate mean at y=1.05 *paper*, anchored left
    fig_hist.add_annotation(
        x=mean_size, y=1.05,
        xref="x", yref="paper",
        text=f"Mean = {mean_size:.2f}",
        showarrow=False,
        font=dict(color="red"),
        xanchor="left", yanchor="bottom"
    )

    # Annotate median at y=1.00 *paper*, anchored right
    fig_hist.add_annotation(
        x=median_size, y=1.00,
        xref="x", yref="paper",
        text=f"Median = {median_size:.2f}",
        showarrow=False,
        font=dict(color="blue"),
        xanchor="right", yanchor="bottom"
    )

    # Tweak axes & styling
    fig_hist.update_layout(
        yaxis_title="Number of Clusters",
        xaxis_title="Cluster Cable Length (μm)",
    )
    fig_hist.update_xaxes(dtick=0.5)
    fig_hist.update_yaxes(dtick=20)
    fig_hist.update_traces(marker_line_color="black", marker_line_width=1)

    # Print stats
    print("Max cluster cable length:   ", cluster_df["minimal_cable_length"].max())
    print("Mean cluster cable length:  ", mean_size)
    print("Median cluster cable length:", median_size)

    # Save to PNG
    if save_plot:
        filename = config.get_filename("cluster_cable_length_distribution", "")
        output_path = config.get_output_path("histo_plots", filename)
        fig_hist.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        # Also save as SVG
        svg_path = output_path.with_suffix('.svg')
        fig_hist.write_image(str(svg_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved excitatory cluster cable length histogram to {output_path}")

    return fig_hist


def plot_inhibitory_density_3d(
    calculation_nodes_inh: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create 3D visualization of inhibitory synapse density.
    
    Args:
        calculation_nodes_inh: DataFrame with inhibitory density nodes
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    # Extract min and max density
    density_min_inh = calculation_nodes_inh['synapse_density'].min()
    density_max_inh = calculation_nodes_inh['synapse_density'].max()

    # Custom colorscale emphasizing top 20%
    custom_colorscale = [
        [0.0, "black"],      # Very low values
        [0.6, "yellow"],     # Before the top
        [1.0, "red"]         # Top 20% clearly highlighted
    ]

    fig_density_inh = go.Figure(layout=dict(
        width=config.figure_width,
        height=config.figure_height,
        title=f"Inhibitory Synapse Density - {config.neuron_id} (After Clustering and Filtering)",
        scene=dict(
            xaxis_title="X (μm)",
            yaxis_title="Y (μm)",
            zaxis_title="Z (μm)",
            aspectmode="data"
        )
    ))

    fig_density_inh.add_trace(go.Scatter3d(
        x=calculation_nodes_inh['x'],
        y=calculation_nodes_inh['y'],
        z=calculation_nodes_inh['z'],
        mode='markers',
        marker=dict(
            size=3,
            color=calculation_nodes_inh['synapse_density'],
            colorscale=custom_colorscale,  
            cmin=density_min_inh,  
            cmax=density_max_inh,  
            colorbar=dict(title="Density"),
            opacity=0.8
        ),
        name="Inhibitory Synapse Density"
    ))

    # Save to file
    if save_plot:
        filename = config.get_filename("inhibitory_density", "")
        output_path = config.get_output_path("histo_plots", filename)
        fig_density_inh.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved inhibitory density visualization to {output_path}")

    return fig_density_inh


def plot_inhibitory_synapse_count_histogram(
    cluster_df_inh: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create histogram of inhibitory cluster synapse counts.
    
    Args:
        cluster_df_inh: DataFrame with inhibitory cluster information
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    fig_hist = px.histogram(
        cluster_df_inh,
        x="i_synapse_count",
        nbins=40,  
        title="Histogram of Inhibitory Cluster Synapse Counts (After Clustering and Filtering)",
        labels={"i_synapse_count": "Number of Synapses per Cluster"},
        text_auto=True
    )
    fig_hist.update_layout(yaxis_title="Number of Clusters")
    fig_hist.update_yaxes(dtick=50)
    fig_hist.update_xaxes(dtick=1)
    fig_hist.update_traces(marker_line_color='black', marker_line_width=1)

    # Save to file
    if save_plot:
        filename = config.get_filename("inhibitory_histo_number_of_syn_per_cluster", "")
        output_path = config.get_output_path("histo_plots", filename)
        fig_hist.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved inhibitory cluster synapse count histogram to {output_path}")

    return fig_hist


def plot_inhibitory_density_histogram(
    cluster_df_inh: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create histogram of inhibitory cluster density distribution.
    
    Args:
        cluster_df_inh: DataFrame with inhibitory cluster information
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    fig_hist = px.histogram(
        cluster_df_inh,
        x="cluster_density",
        nbins=50,  
        title="Distribution of Inhibitory Cluster Density (After Clustering and Filtering)",
        labels={"cluster_density": "Cluster Density"},
        text_auto=True
    )
    fig_hist.update_layout(yaxis_title="Number of Clusters")
    fig_hist.update_yaxes(dtick=40)
    fig_hist.update_xaxes(dtick=1)
    fig_hist.update_traces(marker_line_color='black', marker_line_width=1)

    # Calculate some stats
    print("Max Density: ", cluster_df_inh["cluster_density"].max())
    print("Density mean: ", cluster_df_inh["cluster_density"].mean())
    print("Density median: ", cluster_df_inh["cluster_density"].median())

    # Save to file
    if save_plot:
        filename = config.get_filename("inhibitory_density_distribution_per_cluster", "")
        output_path = config.get_output_path("histo_plots", filename)
        fig_hist.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved inhibitory cluster density histogram to {output_path}")

    return fig_hist


def plot_e_i_distance_histogram(
    all_synapses: pd.DataFrame,
    dynamical_cutoff: float,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create histogram of E-I synapse distances within clusters.
    
    Args:
        all_synapses: Combined DataFrame of inhibitory synapses (clustered and filtered out)
        dynamical_cutoff: Cutoff threshold for distance filtering
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    # Create histogram
    fig = px.histogram(
        all_synapses,
        x="min_dist_e_syn_in_clu",
        nbins=100,
        range_x=[0, 10],
        labels={"min_dist_e_syn_in_clu": "min_dist_e_syn_in_clu (μm)"},
        title="Distribution of Inhibitory Synapse Distances to Excitatory Synapses (After Clustering and Filtering)",
        text_auto=True
    )
    
    # Tweak axes and rename Y
    fig.update_layout(
        bargap=0.05,
        xaxis=dict(
            tick0=0,
            dtick=0.5,
            title_text="Minimal distance of I synapse to E synapse (in cluster) (μm)"
        )
    )

    # Add cutoff line
    fig.add_vline(
        x=dynamical_cutoff,
        line_dash="dash",
        line_color="red",
        annotation_text=f"dynamical cutoff = {dynamical_cutoff:.3f}",
        annotation_position="top right"
    )

    # Save to file
    if save_plot:
        filename = config.get_filename("distribution_of_i_syn_dist_in_cluster", "")
        output_path = config.get_output_path("histo_plots", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        # Also save as SVG
        svg_path = output_path.with_suffix('.svg')
        fig.write_image(str(svg_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved E-I distance histogram to {output_path}")
    
    return fig


def create_excitatory_histograms(
    cluster_df: pd.DataFrame,
    config: VisualizationConfig,
    save_plots: bool = True
) -> dict:
    """
    Create all available excitatory histogram visualizations.
    
    Args:
        cluster_df: DataFrame with excitatory cluster information
        config: Visualization configuration
        save_plots: Whether to save all plots to files
        
    Returns:
        Dictionary of created excitatory figures
    """
    print("Creating excitatory histogram visualizations...")
    
    figures = {}
    
    # Excitatory cluster histograms
    figures['excitatory_synapse_count'] = plot_excitatory_synapse_count_histogram(
        cluster_df, config, save_plots
    )
    
    figures['excitatory_density'] = plot_excitatory_density_histogram(
        cluster_df, config, save_plots
    )
    
    figures['excitatory_cable_length'] = plot_excitatory_cable_length_histogram(
        cluster_df, config, save_plots
    )
    
    print(f"Created {len(figures)} excitatory histogram visualizations successfully!")
    print(f"Excitatory histograms saved to: {config.get_output_path('histo_plots', '')}")
    
    return figures


def create_all_histograms(
    cluster_df: pd.DataFrame,
    cluster_df_inh: pd.DataFrame,
    calculation_nodes_inh: pd.DataFrame,
    all_synapses: pd.DataFrame,
    dynamical_cutoff: float,
    config: VisualizationConfig,
    save_plots: bool = True
) -> dict:
    """
    Create all histogram visualizations for the analysis.
    
    Args:
        cluster_df: DataFrame with excitatory cluster information
        cluster_df_inh: DataFrame with inhibitory cluster information
        calculation_nodes_inh: DataFrame with inhibitory density nodes
        all_synapses: Combined DataFrame of inhibitory synapses
        dynamical_cutoff: Cutoff threshold for distance filtering
        config: Visualization configuration
        save_plots: Whether to save all plots to files
        
    Returns:
        Dictionary of all created figures
    """
    print("Creating all histogram visualizations...")
    
    figures = {}
    
    # Excitatory cluster histograms
    figures['excitatory_synapse_count'] = plot_excitatory_synapse_count_histogram(
        cluster_df, config, save_plots
    )
    
    figures['excitatory_density'] = plot_excitatory_density_histogram(
        cluster_df, config, save_plots
    )
    
    figures['excitatory_cable_length'] = plot_excitatory_cable_length_histogram(
        cluster_df, config, save_plots
    )
    
    # Inhibitory visualizations
    figures['inhibitory_density_3d'] = plot_inhibitory_density_3d(
        calculation_nodes_inh, config, save_plots
    )
    
    figures['inhibitory_synapse_count'] = plot_inhibitory_synapse_count_histogram(
        cluster_df_inh, config, save_plots
    )
    
    figures['inhibitory_density'] = plot_inhibitory_density_histogram(
        cluster_df_inh, config, save_plots
    )
    
    # E-I distance histogram
    figures['e_i_distance'] = plot_e_i_distance_histogram(
        all_synapses, dynamical_cutoff, config, save_plots
    )
    
    print(f"Created {len(figures)} histogram visualizations successfully!")
    print(f"All histograms saved to: {config.get_output_path('histo_plots', '')}")
    
    return figures
