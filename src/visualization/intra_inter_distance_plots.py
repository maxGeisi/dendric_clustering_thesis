# src/visualization/intra_inter_distance_plots.py
"""
Visualization functions for intra-cluster vs inter-cluster distance analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import wilcoxon
from typing import Optional
from pathlib import Path

from .visualization_config import VisualizationConfig


def get_significance_star(p: float) -> str:
    """Get significance star based on p-value."""
    if p <= 0.001:
        return '***'
    elif p <= 0.01:
        return '**'
    elif p <= 0.05:
        return '*'
    else:
        return 'ns'


def plot_intra_vs_inter_cluster_distances(cluster_df: pd.DataFrame,
                                          config: VisualizationConfig,
                                          save_plot: bool = True) -> go.Figure:
    print(f"n clusters total: {len(cluster_df)}")

    intra_col = 'median_intra_dist' if 'median_intra_dist' in cluster_df.columns else 'median_intra_dist_intra'
    inter_col = 'inter_median_dist' if 'inter_median_dist' in cluster_df.columns else 'inter_median_dist_inter'

    paired = cluster_df.dropna(subset=[intra_col, inter_col])
    print(f"n clusters paired: {len(paired)}")
    if len(paired) == 0:
        return go.Figure()

    # Wilcoxon (uncomment if you want stats in the title)
    stat, p_wil = wilcoxon(paired[intra_col], paired[inter_col], alternative='less')
    stars = get_significance_star(p_wil)

    # Decide candidate ID column
    candidate_id = 'e_cluster_id' if 'e_cluster_id' in paired.columns else (
        'cluster_id' if 'cluster_id' in paired.columns else None
    )
    print(f"Using cluster ID candidate: {candidate_id}")
    print(f"Available columns: {list(paired.columns)}")

    # After reset_index, build id_vars ONLY with cols that truly exist
    dfm = paired.reset_index()
    id_vars = []
    for c in ['index', candidate_id]:
        if c is not None and c in dfm.columns:
            id_vars.append(c)

    # Melt safely
    dfm = dfm.melt(
        id_vars=id_vars,
        value_vars=[intra_col, inter_col],
        var_name='Type',
        value_name='Distance'
    )
    dfm['Type'] = dfm['Type'].map({
        'median_intra_dist': 'Within-cluster NN',
        'median_intra_dist_intra': 'Within-cluster NN',
        'inter_median_dist': 'Between-cluster NN',
        'inter_median_dist_inter': 'Between-cluster NN'
    })

    fig = px.violin(
        dfm, x='Type', y='Distance', color='Type',
        box=True, points='all',
        color_discrete_map={'Within-cluster NN': 'steelblue', 'Between-cluster NN': 'orange'},
        width=700, height=550,
        title=(f"Cluster-level NN distances: Within vs Between"
               f"<br><sub>Wilcoxon: W={stat:.1f}, p={p_wil:.3e} {stars}</sub>")
    )
    fig.update_traces(meanline_visible=True, width=0.6, opacity=0.6)
    
    # Annotate mean & median under each violin
    stats_df = dfm.groupby('Type')['Distance'].agg(mean='mean', median='median').reset_index()
    for _, row in stats_df.iterrows():
        fig.add_annotation(
            x=row['Type'], 
            yref='paper', 
            y=-0.24,
            text=f"μ={row['mean']:.2f}<br>med={row['median']:.2f}",
            showarrow=False,
            font=dict(size=12),
            align='center'
        )
    
    # Add significance bar + stars
    y_max = dfm['Distance'].max()
    y_sig = y_max * 1.05
    
    # horizontal bar
    fig.add_shape(
        type="line",
        xref="x", 
        yref="y",
        x0="Within-cluster NN", 
        x1="Between-cluster NN",
        y0=y_sig, 
        y1=y_sig,
        line=dict(color="black", width=1)
    )
    # stars
    fig.add_annotation(
        x=0.5, 
        xref="paper",
        y=y_sig * 1.02, 
        yref="y",
        text=stars,
        showarrow=False,
        font=dict(size=20)
    )
    
    # Final layout tweaks
    fig.update_layout(
        height=700, 
        width=900,
        margin=dict(t=100, b=160, l=60, r=20),
        yaxis=dict(title="Nearest-neighbor distance (μm)", rangemode="nonnegative"),
        xaxis=dict(title=""),
        showlegend=False
    )
    
    # Save plot if requested
    if save_plot:
        filename = config.get_filename("cluster_nn_distances", "violin")
        output_path = config.get_output_path("clusters", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved intra vs inter cluster distance plot: {output_path}")
    
    return fig


def plot_intra_cluster_distance_histogram(
    cluster_df: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create histogram of intra-cluster distances with mean and max values.
    
    Args:
        cluster_df: DataFrame with cluster information including intra distances
        config: Visualization configuration
        save_plot: Whether to save the plot
        
    Returns:
        Plotly figure object
    """
    # Filter out clusters with no intra-cluster distances
    # Check for both possible column name patterns
    intra_col = 'median_intra_dist' if 'median_intra_dist' in cluster_df.columns else 'median_intra_dist_intra'
    
    valid_data = cluster_df[cluster_df[intra_col] > 0].copy()
    
    if len(valid_data) == 0:
        print("No clusters with valid intra-cluster distances")
        return go.Figure()
    
    # Create histogram
    fig = px.histogram(
        valid_data,
        x=intra_col,
        nbins=30,
        title=f"Distribution of Intra-cluster Distances<br><sub>Mean: {valid_data[intra_col].mean():.2f} μm, Max: {valid_data[intra_col].max():.2f} μm</sub>",
        labels={intra_col: 'Median Intra-cluster Distance (μm)', 'count': 'Number of Clusters'}
    )
    
    # Add vertical lines for mean and max
    mean_val = valid_data[intra_col].mean()
    max_val = valid_data[intra_col].max()
    
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.2f} μm"
    )
    
    fig.add_vline(
        x=max_val,
        line_dash="dot",
        line_color="blue",
        annotation_text=f"Max: {max_val:.2f} μm"
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=800,
        margin=dict(t=80, b=60, l=60, r=20)
    )
    
    # Save plot if requested
    if save_plot:
        filename = config.get_filename("intra_cluster_distances", "histogram")
        output_path = config.get_output_path("clusters", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved intra-cluster distance histogram: {output_path}")
    
    return fig


def plot_inter_cluster_distance_histogram(
    cluster_df: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create histogram of inter-cluster distances with mean and max values.
    
    Args:
        cluster_df: DataFrame with cluster information including inter distances
        config: Visualization configuration
        save_plot: Whether to save the plot
        
    Returns:
        Plotly figure object
    """
    # Filter out clusters with no inter-cluster distances
    # Check for both possible column name patterns
    inter_col = 'inter_median_dist' if 'inter_median_dist' in cluster_df.columns else 'inter_median_dist_inter'
    
    valid_data = cluster_df.dropna(subset=[inter_col]).copy()
    
    if len(valid_data) == 0:
        print("No clusters with valid inter-cluster distances")
        return go.Figure()
    
    # Create histogram
    fig = px.histogram(
        valid_data,
        x=inter_col,
        nbins=30,
        title=f"Distribution of Inter-cluster Distances<br><sub>Mean: {valid_data[inter_col].mean():.2f} μm, Max: {valid_data[inter_col].max():.2f} μm</sub>",
        labels={inter_col: 'Median Inter-cluster Distance (μm)', 'count': 'Number of Clusters'}
    )
    
    # Add vertical lines for mean and max
    mean_val = valid_data[inter_col].mean()
    max_val = valid_data[inter_col].max()
    
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.2f} μm"
    )
    
    fig.add_vline(
        x=max_val,
        line_dash="dot",
        line_color="blue",
        annotation_text=f"Max: {max_val:.2f} μm"
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=800,
        margin=dict(t=80, b=60, l=60, r=20)
    )
    
    # Save plot if requested
    if save_plot:
        filename = config.get_filename("inter_cluster_distances", "histogram")
        output_path = config.get_output_path("clusters", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved inter-cluster distance histogram: {output_path}")
    
    return fig
