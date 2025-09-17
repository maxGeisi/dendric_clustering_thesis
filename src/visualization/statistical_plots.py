# src/visualization/statistical_plots.py
"""
Statistical analysis visualization functions for cluster properties.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu
from pathlib import Path
from typing import Optional

from .visualization_config import VisualizationConfig


def get_significance_star(p: float) -> str:
    """
    Get significance star annotation based on p-value.
    
    Args:
        p: P-value from statistical test
        
    Returns:
        String with appropriate significance symbol
    """
    if p <= 0.001:
        return '***'
    elif p <= 0.01:
        return '**'
    elif p <= 0.05:
        return '*'
    else:
        return 'ns'


def plot_cluster_cable_length_comparison(
    cluster_df: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create violin plot comparing minimal cable length between clusters with and without I synapses.
    
    Args:
        cluster_df: DataFrame with cluster information including minimal_cable_length and has_I_associated
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    # Prepare data
    df = cluster_df.copy()
    df['I_group'] = df['has_I_associated'].map({False: 'No I', True: 'With I'})
    
    # Extract the two groups
    no_cable = df.loc[df["I_group"] == "No I", "minimal_cable_length"]
    wi_cable = df.loc[df["I_group"] == "With I", "minimal_cable_length"]
    
    # Mann–Whitney U test
    _, p_val = mannwhitneyu(no_cable, wi_cable, alternative="two-sided")
    star = get_significance_star(p_val)
    
    # Create violin plot with p-value in title
    title_with_p = f"Excitatory Cluster Cable Length: With vs Without Inhibitory Synapses<br><sub>Mann-Whitney U test: p = {p_val:.4f}</sub>"
    
    fig = px.violin(
        df,
        x="I_group",
        y="minimal_cable_length",
        color="I_group",
        box=True,
        points=False,
        title=title_with_p,
        labels={"I_group": "", "minimal_cable_length": "Minimal Cable Length (μm)"}
    )
    
    # Add significance bracket and star
    ymax = df["minimal_cable_length"].max()
    bracket = ymax * 1.05
    star_y = ymax * 1.08
    
    fig.add_shape(
        type="line",
        xref="paper", x0=0.25, x1=0.75,
        yref="y", y0=bracket, y1=bracket,
        line=dict(color="black", width=1.5)
    )
    fig.add_annotation(
        xref="paper", x=0.5,
        yref="y", y=star_y,
        text=star,
        showarrow=False,
        font=dict(size=18),
        yanchor="bottom"
    )
    
    # Annotate mean & median under each violin
    stats_df = (
        df.groupby("I_group")["minimal_cable_length"]
        .agg(mean="mean", median="median")
        .reset_index()
    )
    for _, row in stats_df.iterrows():
        fig.add_annotation(
            x=row["I_group"],
            y=-0.15,  # Positioned below plot with more space
            xref="x", yref="paper",
            text=f"μ={row['mean']:.1f}<br>med={row['median']:.1f}",
            showarrow=False,
            font=dict(size=12),
            align="center"
        )
    
    # Final layout tweaks with more space for annotations
    fig.update_layout(
        yaxis=dict(title="Minimal Cable Length (μm)"),
        margin=dict(t=80, b=120),  # Increased top and bottom margins
        showlegend=False
    )
    
    # Save plot if requested
    if save_plot:
        filename = config.get_filename("excitatory_cable_length", "with_vs_without_inhibitory")
        output_path = config.get_output_path("t_test_violin", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved cable length comparison: {output_path}")
    
    return fig


def plot_cluster_synapse_count_comparison(
    cluster_df: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create violin plot comparing E-synapse count between clusters with and without I synapses.
    
    Args:
        cluster_df: DataFrame with cluster information including e_synapse_count and has_I_associated
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    # Prepare data
    df = cluster_df.copy()
    df['I_group'] = df['has_I_associated'].map({False: 'No I', True: 'With I'})
    
    # Extract the two groups
    no_ec = df.loc[df["I_group"] == "No I", "e_synapse_count"]
    wi_ec = df.loc[df["I_group"] == "With I", "e_synapse_count"]
    
    # Mann–Whitney U test
    _, p_val = mannwhitneyu(no_ec, wi_ec, alternative="two-sided")
    star = get_significance_star(p_val)
    
    # Create violin plot with p-value in title
    title_with_p = f"Excitatory Synapse Count: With vs Without Inhibitory Synapses<br><sub>Mann-Whitney U test: p = {p_val:.4f}</sub>"
    
    fig = px.violin(
        df,
        x="I_group",
        y="e_synapse_count",
        color="I_group",
        box=True,
        points=False,
        title=title_with_p,
        labels={"I_group": "", "e_synapse_count": "# E Synapses per Cluster"}
    )
    
    # Add significance bracket and star
    ymax = df["e_synapse_count"].max()
    bracket = ymax * 1.05
    star_y = ymax * 1.08
    
    fig.add_shape(
        type="line",
        xref="paper", x0=0.25, x1=0.75,
        yref="y", y0=bracket, y1=bracket,
        line=dict(color="black", width=1.5)
    )
    fig.add_annotation(
        xref="paper", x=0.5,
        yref="y", y=star_y,
        text=star,
        showarrow=False,
        font=dict(size=18),
        yanchor="bottom"
    )
    
    # Annotate mean & median under each violin
    stats_df = (
        df.groupby("I_group")["e_synapse_count"]
        .agg(mean="mean", median="median")
        .reset_index()
    )
    for _, row in stats_df.iterrows():
        fig.add_annotation(
            x=row["I_group"],
            y=-0.15,  # Positioned below plot with more space
            xref="x", yref="paper",
            text=f"μ={row['mean']:.1f}<br>med={row['median']:.1f}",
            showarrow=False,
            font=dict(size=12),
            align="center"
        )
    
    # Final layout tweaks with more space for annotations
    fig.update_layout(
        yaxis=dict(title="# E Synapses"),
        margin=dict(t=80, b=120),  # Increased top and bottom margins
        showlegend=False
    )
    
    # Save plot if requested
    if save_plot:
        filename = config.get_filename("excitatory_synapse_count", "with_vs_without_inhibitory")
        output_path = config.get_output_path("t_test_violin", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved synapse count comparison: {output_path}")
    
    return fig


def plot_cluster_density_comparison(
    cluster_df: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create violin plot comparing cluster density between clusters with and without I synapses.
    
    Args:
        cluster_df: DataFrame with cluster information including cluster_density and has_I_associated
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    # Prepare data
    df = cluster_df.copy()
    df['I_group'] = df['has_I_associated'].map({False: 'No I', True: 'With I'})
    
    # Extract the two groups
    no_d = df.loc[df['I_group'] == 'No I', 'cluster_density']
    wi_d = df.loc[df['I_group'] == 'With I', 'cluster_density']
    
    # Mann–Whitney U test
    u_stat, p_val = mannwhitneyu(no_d, wi_d, alternative='two-sided')
    star = get_significance_star(p_val)
    
    # Create violin plot with p-value in title
    title_with_p = f'Excitatory Cluster Density: With vs Without Inhibitory Synapses<br><sub>Mann-Whitney U test: p = {p_val:.4f}</sub>'
    
    fig = px.violin(
        df,
        x='I_group',
        y='cluster_density',
        color='I_group',
        box=True,
        points=False,
        title=title_with_p,
        labels={'I_group': '', 'cluster_density': 'Cluster Density (synapses/μm)'}
    )
    
    # Add significance bracket and star
    ymax = df['cluster_density'].max()
    bracket_y = ymax * 1.05
    star_y = ymax * 1.08
    
    fig.add_shape(
        type='line',
        xref='paper', x0=0.25, x1=0.75,
        yref='y', y0=bracket_y, y1=bracket_y,
        line=dict(color='black', width=1.5)
    )
    fig.add_annotation(
        xref='paper', x=0.5,
        yref='y', y=star_y,
        text=star,
        showarrow=False,
        font=dict(size=20, color='black'),
        yanchor='bottom'
    )
    
    # Annotate mean and median below each violin
    stats_df = (
        df.groupby('I_group')['cluster_density']
        .agg(mean='mean', median='median')
        .reset_index()
    )
    for _, row in stats_df.iterrows():
        fig.add_annotation(
            x=row['I_group'],
            y=-0.15,  # Positioned below plot with more space
            xref='x', yref='paper',
            text=f"μ={row['mean']:.2f}<br>med={row['median']:.2f}",
            showarrow=False,
            font=dict(size=12),
            align='center'
        )
    
    # Final layout tweaks with more space for annotations
    fig.update_layout(
        yaxis=dict(title='Cluster Density (synapses/μm)'),
        margin=dict(t=80, b=120),  # Increased top and bottom margins
        showlegend=False
    )
    
    # Save plot if requested
    if save_plot:
        filename = config.get_filename("excitatory_cluster_density", "with_vs_without_inhibitory")
        output_path = config.get_output_path("t_test_violin", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved cluster density comparison: {output_path}")
    
    return fig


def create_all_statistical_plots(
    cluster_df: pd.DataFrame,
    config: VisualizationConfig,
    save_plots: bool = True
) -> dict:
    """
    Create all statistical comparison plots for cluster properties.
    
    Args:
        cluster_df: DataFrame with cluster information
        config: Visualization configuration
        save_plots: Whether to save plots to files
        
    Returns:
        Dictionary of created figures
    """
    figures = {}
    
    # Check if required columns exist
    required_cols = ['minimal_cable_length', 'e_synapse_count', 'cluster_density', 'has_I_associated']
    missing_cols = [col for col in required_cols if col not in cluster_df.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns for statistical plots: {missing_cols}")
        return figures
    
    # Create cable length comparison
    figures['cable_length_comparison'] = plot_cluster_cable_length_comparison(
        cluster_df, config, save_plots
    )
    
    # Create synapse count comparison
    figures['synapse_count_comparison'] = plot_cluster_synapse_count_comparison(
        cluster_df, config, save_plots
    )
    
    # Create density comparison
    figures['density_comparison'] = plot_cluster_density_comparison(
        cluster_df, config, save_plots
    )
    
    print(f"\nCreated {len(figures)} statistical comparison plots successfully!")
    
    return figures
