# src/visualization/branch_plots.py
"""
Branch visualization functions for dendric clustering analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.colors import qualitative as q
from scipy.stats import pearsonr, spearmanr, gaussian_kde
import matplotlib.pyplot as plt
import navis
from typing import Dict, List, Any, Optional
from pathlib import Path

from .visualization_config import VisualizationConfig


def plot_branch_length_distribution(
    neuron_splits: List[Any],
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Plot distribution of branch cable lengths (geodesic) with median and mean lines.
    
    Args:
        neuron_splits: List of branch skeleton objects
        config: Visualization configuration
        save_plot: Whether to save the plot
        
    Returns:
        Plotly figure
    """
    # 1) Extract branch lengths and enforce float dtype
    branch_lengths = [float(branch.cable_length) for branch in neuron_splits]
    df_len = pd.DataFrame({'branch_length': branch_lengths}, dtype='float64')
    
    # 2) Compute median & mean
    median_len = np.median(branch_lengths)
    mean_len = np.mean(branch_lengths)
    
    # 3) Build the histogram
    fig = px.histogram(
        df_len,
        x='branch_length',
        nbins=50,
        text_auto=True,  # show count on each bar
        labels={'branch_length': 'Branch length (μm)'},
        width=1000, 
        height=500,
        title='Distribution of Branch Cable Lengths (geodesic)'
    )
    
    # 4) Add median line + label
    fig.add_shape(
        type='line',
        x0=median_len, x1=median_len,
        y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color='red', dash='dash', width=2)
    )
    fig.add_annotation(
        x=median_len, y=1.02,
        xref='x', yref='paper',
        text=f'Median = {median_len:.2f} μm',
        showarrow=False,
        font=dict(color='red')
    )
    
    # 5) Add mean line + label
    fig.add_shape(
        type='line',
        x0=mean_len, x1=mean_len,
        y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color='blue', dash='dot', width=2)
    )
    fig.add_annotation(
        x=mean_len, y=1.06,
        xref='x', yref='paper',
        text=f'Mean = {mean_len:.2f} μm',
        showarrow=False,
        font=dict(color='blue')
    )
    
    # 6) Tidy up
    fig.update_layout(
        margin=dict(t=80, b=80, l=60, r=20)
    )
    
    if save_plot:
        output_path = config.base_output_dir / "branches" / f"{config.neuron_id}_branch_length_distribution.{config.format}"
        fig.write_image(str(output_path), width=1000, height=500, scale=2)
        print(f"Branch length distribution plot saved to: {output_path}")
    
    return fig


def plot_branch_points_3d(
    neuron_skel,
    junctions_list: List[int],
    calculation_nodes: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Plot 3D visualization of neuron skeleton with branch points highlighted.
    
    Args:
        neuron_skel: Neuron skeleton object
        junctions_list: List of junction node IDs
        calculation_nodes: DataFrame with node coordinates
        config: Visualization configuration
        save_plot: Whether to save the plot
        
    Returns:
        Plotly figure
    """
    # 2) Look up 3D coordinates of junction nodes
    jn = calculation_nodes.set_index("node_id").loc[junctions_list]
    
    # 3) Build 3D figure
    fig = go.Figure(layout=dict(width=1200, height=800))
    
    # a) Draw the full skeleton in light grey (using navis)
    import navis
    navis.plot3d(
        neuron_skel,
        fig=fig,
        color="green",
        palette="viridis",
        alpha=0.2,
        legend=False
    )
    
    # b) Overlay the branch-points as red spheres
    fig.add_trace(go.Scatter3d(
        x=jn["x"],
        y=jn["y"],
        z=jn["z"],
        mode="markers",
        marker=dict(size=2, color="red"),
        name="Branch points"
    ))
    
    # 4) Zoom to extents + show
    fig.update_layout(
        scene=dict(aspectmode="data"),
        title="Neuron + branch-point nodes"
    )
    
    if save_plot:
        output_path = config.base_output_dir / "branches" / f"{config.neuron_id}_branch_points_3d.{config.format}"
        fig.write_image(str(output_path), width=1200, height=800, scale=2)
        print(f"Branch points 3D plot saved to: {output_path}")
    
    return fig


def plot_branch_volume_correlation(
    branch_df: pd.DataFrame,
    syn_exec_df: pd.DataFrame,
    syn_inh_df: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Plot branch-by-branch E vs I synapse volume correlation with fit line and statistics.
    
    Args:
        branch_df: DataFrame with branch information
        syn_exec_df: DataFrame with excitatory synapse data
        syn_inh_df: DataFrame with inhibitory synapse data
        config: Visualization configuration
        save_plot: Whether to save the plot
        
    Returns:
        Plotly figure
    """
    # Determine the correct column names
    exec_id_col = 'Exec_syn_id' if 'Exec_syn_id' in syn_exec_df.columns else 'id'
    inh_id_col = 'Inh_syn_id' if 'Inh_syn_id' in syn_inh_df.columns else 'id'
    
    # 1) Sum volumes per branch
    branch_df = branch_df.copy()
    branch_df['exec_vol_sum'] = branch_df['e_syn_ids'].apply(
        lambda ids: syn_exec_df.loc[
            syn_exec_df[exec_id_col].isin(ids), 'SynapseVolume'
        ].sum() if ids else 0
    )
    branch_df['inh_vol_sum'] = branch_df['i_syn_ids'].apply(
        lambda ids: syn_inh_df.loc[
            syn_inh_df[inh_id_col].isin(ids), 'SynapseVolume'
        ].sum() if ids else 0
    )
    branch_df['total_syn'] = branch_df['n_e'] + branch_df['n_i']
    
    # 2) Prepare x & y (E‐volume vs. I‐volume)
    x = branch_df['exec_vol_sum'].values   # Total E synapse volume
    y = branch_df['inh_vol_sum'].values    # Total I synapse volume
    
    # 3) Fit + correlations
    slope, intercept = np.polyfit(x, y, 1)
    r, p_pearson = pearsonr(x, y)
    rho, p_spearman = spearmanr(x, y)
    
    # 4) Fit line endpoints
    x_line = np.array([0, x.max()])
    y_line = intercept + slope * x_line
    
    # 5) Make the bubble‐scatter with E on x and I on y
    fig = px.scatter(
        branch_df,
        x='exec_vol_sum',
        y='inh_vol_sum',
        size='total_syn',
        color='total_syn',
        color_continuous_scale='Viridis',
        size_max=40,
        labels={
            'exec_vol_sum': 'Total E synapse volume (µm³)',
            'inh_vol_sum': 'Total I synapse volume (µm³)',
            'total_syn': 'Total synapses'
        },
        width=850,
        height=600,
        title='Branch‐by‐branch: E vs I Synapse Volume'
    )
    
    # 6) Hide the bubble trace from the legend
    fig.update_traces(selector=dict(mode='markers'), showlegend=False)
    
    # 7) Overlay the fit line (no legend entry)
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        showlegend=False
    ))
    
    # 8) Embed fit stats inside the plot
    fig.add_annotation(
        x=0.65, y=0.95,
        xref='paper', yref='paper',
        text=(
            f"<b>Fit:</b> I = {slope:.2f}·E + {intercept:.2f}<br>"
            f"<b>Pearson:</b> r={r:.2f} (p={p_pearson:.3f})<br>"
            f"<b>Spearman:</b> rho={rho:.2f} (p={p_spearman:.3f})"
        ),
        align='left',
        showarrow=False,
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1,
        font=dict(size=12)
    )
    
    # 9) Tidy axes and colorbar
    fig.update_xaxes(title='Total E synapse volume (µm³)')
    fig.update_yaxes(title='Total I synapse volume (µm³)')
    fig.update_layout(margin=dict(l=60, r=80, t=80, b=60))
    fig.update_coloraxes(colorbar_x=1.02, colorbar_len=0.8)
    
    if save_plot:
        output_path = config.base_output_dir / "branches" / f"{config.neuron_id}_branch_volume_correlation.{config.format}"
        fig.write_image(str(output_path), width=850, height=600, scale=2)
        print(f"Branch volume correlation plot saved to: {output_path}")
    
    return fig


def plot_branch_synapse_count_correlation(
    branch_df: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Plot branch-by-branch I-synapses vs E-synapses with linear fit.
    
    Args:
        branch_df: DataFrame with branch information
        config: Visualization configuration
        save_plot: Whether to save the plot
        
    Returns:
        Plotly figure
    """
    # Prepare x & y
    x = branch_df['n_i'].values
    y = branch_df['n_e'].values
    
    # 1) Fit a simple linear regression: y = m·x + b
    m, b = np.polyfit(x, y, 1)
    
    # 2) Define the fit line endpoints
    x0, x1 = 0, x.max() + 1
    y0, y1 = m * x0 + b, m * x1 + b
    
    # 3) Build the scatter + fit‐line figure
    fig = go.Figure()
    
    # scatter of branches
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=6, color='blue', opacity=0.7),
        name='Branches'
    ))
    
    # red fit line
    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1],
        mode='lines',
        line=dict(color='red', width=2),
        name=f'Fit: y={m:.2f}x+{b:.2f}'
    ))
    
    # 4) Layout tweaks
    fig.update_layout(
        title='Branch‐by‐branch: # I-synapses vs # E-synapses',
        xaxis_title='# I-synapses',
        yaxis_title='# E-synapses',
        width=900,
        height=700,
    )
    
    # 5) Force axes to start at zero
    fig.update_xaxes(range=[0, x1])
    fig.update_yaxes(range=[0, max(y.max(), y0, y1) + 1])
    
    if save_plot:
        output_path = config.base_output_dir / "branches" / f"{config.neuron_id}_branch_synapse_count_correlation.{config.format}"
        fig.write_image(str(output_path), width=900, height=700, scale=2)
        print(f"Branch synapse count correlation plot saved to: {output_path}")
    
    return fig


def plot_branch_filtering_summary(
    filter_stats: Dict[str, Any],
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Plot summary of branch filtering results.
    
    Args:
        filter_stats: Dictionary with filtering statistics
        config: Visualization configuration
        save_plot: Whether to save the plot
        
    Returns:
        Plotly figure
    """
    # Create subplots
    fig = go.Figure()
    
    # Create a simple summary plot
    categories = ['Branches Kept', 'Branches Removed', 'E-synapses Lost', 'I-synapses Lost']
    values = [
        filter_stats['total_branches'] - filter_stats['branches_removed'],
        filter_stats['branches_removed'],
        filter_stats['e_synapses_lost'],
        filter_stats['i_synapses_lost']
    ]
    colors = ['green', 'red', 'orange', 'purple']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=values,
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f"Branch Filtering Summary - {config.neuron_id}<br>"
              f"5th-percentile cutoff: {filter_stats['cutoff']:.2f} μm",
        xaxis_title="Category",
        yaxis_title="Count",
        width=800,
        height=500
    )
    
    if save_plot:
        output_path = config.base_output_dir / "branches" / f"{config.neuron_id}_branch_filtering_summary.{config.format}"
        fig.write_image(str(output_path), width=800, height=500, scale=2)
        print(f"Branch filtering summary plot saved to: {output_path}")
    
    return fig


def plot_branch_synapses_3d(
    branch_idx: int,
    neuron_splits: List[Any],
    syn_exec_df: pd.DataFrame,
    syn_inh_df: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Plot 3D visualization of synapses on a specific branch.
    
    Args:
        branch_idx: Index of the branch to visualize
        neuron_splits: List of branch skeleton objects
        syn_exec_df: DataFrame with excitatory synapse data
        syn_inh_df: DataFrame with inhibitory synapse data
        config: Visualization configuration
        save_plot: Whether to save the plot
        
    Returns:
        Plotly figure
    """
    # Select the branch and its synapses
    rand_branch = neuron_splits[branch_idx]
    node_ids = rand_branch.nodes["node_id"].tolist()
    
    # Get synapses on this branch
    df_e_branch = syn_exec_df[syn_exec_df["closest_node_id"].isin(node_ids)]
    df_i_branch = syn_inh_df[syn_inh_df["closest_node_id"].isin(node_ids)]
    
    # Create figure
    fig = go.Figure()
    
    # Add E-synapses (blue circles)
    fig.add_trace(go.Scatter3d(
        x=df_e_branch["Epos3DX"],
        y=df_e_branch["Epos3DY"],
        z=df_e_branch["Epos3DZ"],
        mode="markers",
        marker=dict(size=5, color="blue", symbol="circle"),
        name="E-synapses"
    ))
    
    # Add I-synapses (red crosses)
    fig.add_trace(go.Scatter3d(
        x=df_i_branch["Ipos3DX"],
        y=df_i_branch["Ipos3DY"],
        z=df_i_branch["Ipos3DZ"],
        mode="markers",
        marker=dict(size=6, color="red", symbol="cross"),
        name="I-synapses"
    ))
    
    # Add skeleton
    start = len(fig.data)
    navis.plot3d(
        rand_branch,
        fig=fig,
        color="green",
        palette="viridis",
        alpha=0.3,
        legend=False,
        inline=False
    )
    end = len(fig.data)
    for i in range(start, end):
        fig.data[i].showlegend = False
    
    fig.update_layout(
        title=f"Branch {branch_idx}: E- (blue) & I- (red) synapses",
        width=800,
        height=600,
        scene=dict(aspectmode="data")
    )
    
    if save_plot:
        output_path = config.base_output_dir / "branches" / f"{config.neuron_id}_branch_{branch_idx}_synapses_3d.{config.format}"
        fig.write_image(str(output_path), width=800, height=600, scale=2)
        print(f"Branch synapses 3D plot saved to: {output_path}")
    
    return fig


def plot_branch_synapses_by_cluster(
    branch_idx: int,
    neuron_splits: List[Any],
    syn_exec_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Plot 3D visualization of excitatory synapses on a branch colored by cluster.
    
    Args:
        branch_idx: Index of the branch to visualize
        neuron_splits: List of branch skeleton objects
        syn_exec_df: DataFrame with excitatory synapse data
        cluster_df: DataFrame with cluster information
        config: Visualization configuration
        save_plot: Whether to save the plot
        
    Returns:
        Plotly figure
    """
    # Select the branch and its synapses
    rand_branch = neuron_splits[branch_idx]
    node_ids = rand_branch.nodes["node_id"].tolist()
    
    # Get synapses on this branch
    df_e_branch = syn_exec_df[syn_exec_df["closest_node_id"].isin(node_ids)]
    
    # Split valid vs invalid clusters
    # Use dynamic column detection for compatibility
    cluster_id_col = 'e_cluster_id' if 'e_cluster_id' in cluster_df.columns else 'cluster_id'
    syn_cluster_id_col = 'cluster_id_exec' if 'cluster_id_exec' in df_e_branch.columns else 'cluster_id'
    
    valid_clusters = set(cluster_df[cluster_id_col])
    df_e_valid = df_e_branch[df_e_branch[syn_cluster_id_col].isin(valid_clusters)]
    df_e_invalid = df_e_branch[~df_e_branch[syn_cluster_id_col].isin(valid_clusters)]
    
    # Create figure
    fig = go.Figure()
    
    # Add valid clusters with different colors
    exec_ids = sorted(df_e_valid[syn_cluster_id_col].unique())
    palette = q.Plotly
    color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(exec_ids)}
    
    for cid, col in color_map.items():
        sub = df_e_valid[df_e_valid[syn_cluster_id_col] == cid]
        fig.add_trace(go.Scatter3d(
            x=sub["Epos3DX"],
            y=sub["Epos3DY"],
            z=sub["Epos3DZ"],
            mode="markers",
            marker=dict(size=5, color=col, symbol="circle"),
            name=f"E-cluster {int(cid)}"
        ))
    
    # Add invalid (filtered-out) synapses in dark grey
    if not df_e_invalid.empty:
        fig.add_trace(go.Scatter3d(
            x=df_e_invalid["Epos3DX"],
            y=df_e_invalid["Epos3DY"],
            z=df_e_invalid["Epos3DZ"],
            mode="markers",
            marker=dict(size=5, color="darkgrey", symbol="circle"),
            name="Unclustered E-synapses"
        ))
    
    # Add skeleton
    start = len(fig.data)
    navis.plot3d(
        rand_branch,
        fig=fig,
        color="green",
        palette="viridis",
        alpha=0.3,
        legend=False,
        inline=False
    )
    end = len(fig.data)
    for i in range(start, end):
        fig.data[i].showlegend = False
    
    fig.update_layout(
        title=f"Branch {branch_idx}: E-synapses colored by valid cluster (grey=filtered out)",
        width=800,
        height=600,
        scene=dict(aspectmode="data")
    )
    
    if save_plot:
        output_path = config.base_output_dir / "branches" / f"{config.neuron_id}_branch_{branch_idx}_synapses_by_cluster.{config.format}"
        fig.write_image(str(output_path), width=800, height=600, scale=2)
        print(f"Branch synapses by cluster plot saved to: {output_path}")
    
    return fig


def plot_synapse_distance_to_soma_histogram(
    syn_exec_df: pd.DataFrame,
    syn_inh_df: pd.DataFrame,
    geodesic_mat_full: pd.DataFrame,
    neuron_skel,
    config: VisualizationConfig,
    save_plot: bool = True
) -> plt.Figure:
    """
    Plot histogram of synapse distances to soma with KDE curves.
    
    Args:
        syn_exec_df: DataFrame with excitatory synapse data
        syn_inh_df: DataFrame with inhibitory synapse data
        geodesic_mat_full: Full geodesic distance matrix
        neuron_skel: Neuron skeleton object
        config: Visualization configuration
        save_plot: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Identify the soma node
    soma_nodes = neuron_skel.nodes.loc[neuron_skel.nodes['parent_id']==-1, 'node_id']
    if soma_nodes.empty:
        raise RuntimeError("No soma node found (parent_id == -1).")
    soma = soma_nodes.iat[0]
    
    # 2) Map each synapse's closest_node_id → geodesic distance from soma
    #    (so we ignore any prior 'distance_to_soma' column and recompute both)
    def lookup_dist(df):
        def safe_lookup(nid):
            try:
                if nid in geodesic_mat_full.index and soma in geodesic_mat_full.columns:
                    dist = geodesic_mat_full.at[soma, nid]
                    # Check if distance is valid
                    if np.isfinite(dist) and dist >= 0:
                        return dist
                    else:
                        return np.nan
                else:
                    return np.nan
            except (KeyError, IndexError):
                return np.nan
        return df['closest_node_id'].map(safe_lookup)
    
    syn_exec_df['dist_um'] = lookup_dist(syn_exec_df)
    syn_inh_df['dist_um'] = lookup_dist(syn_inh_df)
    
    # Debug information
    print(f"Debug: E-synapses total: {len(syn_exec_df)}, I-synapses total: {len(syn_inh_df)}")
    print(f"Debug: E-synapses with valid distances: {syn_exec_df['dist_um'].notna().sum()}")
    print(f"Debug: I-synapses with valid distances: {syn_inh_df['dist_um'].notna().sum()}")
    
    # drop any NaNs and filter out invalid values
    e_d = syn_exec_df['dist_um'].dropna()
    i_d = syn_inh_df['dist_um'].dropna()
    
    # Remove infinite and negative values
    e_d = e_d[np.isfinite(e_d) & (e_d >= 0)].values
    i_d = i_d[np.isfinite(i_d) & (i_d >= 0)].values
    
    print(f"Debug: E-synapses after filtering: {len(e_d)}, I-synapses after filtering: {len(i_d)}")
    
    # Check if we have valid data
    if len(e_d) == 0 and len(i_d) == 0:
        print("Warning: No valid distance data found for synapses")
        # Create empty plot
        plt.figure(figsize=(8, 4))
        plt.xlabel("Distance to soma (µm)")
        plt.ylabel("Normalized number of synapses")
        plt.title("E vs. I synapse distance to soma (No valid data)")
        plt.tight_layout()
        return plt.gcf()
    
    # 3) Prepare common histogram + KDE grid
    xmin, xmax = 0, max(e_d.max(), i_d.max()) * 1.05
    bins = np.linspace(xmin, xmax, 50)
    xgrid = np.linspace(xmin, xmax, 500)
    
    # Calculate KDE with error handling
    kde_e = None
    kde_i = None
    
    try:
        if len(e_d) > 1:  # Need at least 2 points for KDE
            kde_e = gaussian_kde(e_d)
    except Exception as e:
        print(f"Warning: Could not create KDE for E-synapses: {e}")
    
    try:
        if len(i_d) > 1:  # Need at least 2 points for KDE
            kde_i = gaussian_kde(i_d)
    except Exception as e:
        print(f"Warning: Could not create KDE for I-synapses: {e}")
    
    # 4) Plot
    plt.figure(figsize=(8, 4))
    
    # histograms (density normalized)
    if len(e_d) > 0:
        plt.hist(e_d, bins=bins, density=True,
                 color='steelblue', alpha=0.6,
                 label=f"{len(e_d)} E synapses")
    if len(i_d) > 0:
        plt.hist(i_d, bins=bins, density=True,
                 color='salmon', alpha=0.6,
                 label=f"{len(i_d)} I synapses")
    
    # KDE curves (only if KDE was successfully created)
    if kde_e is not None:
        try:
            plt.plot(xgrid, kde_e(xgrid), color='navy', lw=2)
        except Exception as e:
            print(f"Warning: Could not plot E-synapse KDE: {e}")
    
    if kde_i is not None:
        try:
            plt.plot(xgrid, kde_i(xgrid), color='red', lw=2)
        except Exception as e:
            print(f"Warning: Could not plot I-synapse KDE: {e}")
    
    plt.xlabel("Distance to soma (µm)")
    plt.ylabel("Normalized number of synapses")
    plt.title("E vs. I synapse distance to soma")
    plt.legend(fontsize='small')
    plt.tight_layout()
    
    if save_plot:
        output_path = config.base_output_dir / "branches" / f"{config.neuron_id}_synapse_distance_to_soma_histogram.{config.format}"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Synapse distance to soma histogram saved to: {output_path}")
    
    return plt.gcf()


def plot_branch_density_analysis(
    branch_idx: int,
    neuron_splits: List[Any],
    syn_exec_df: pd.DataFrame,
    calculation_nodes: pd.DataFrame,
    geodesic_mat_full: pd.DataFrame,
    config: VisualizationConfig,
    save_plot: bool = True
) -> plt.Figure:
    """
    Plot E vs I density analysis along a branch with synapse rug plot.
    
    Args:
        branch_idx: Index of the branch to analyze
        neuron_splits: List of branch skeleton objects
        syn_exec_df: DataFrame with excitatory synapse data
        calculation_nodes: DataFrame with density calculation nodes
        geodesic_mat_full: Full geodesic distance matrix
        config: Visualization configuration
        save_plot: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Get branch skeleton
    sk = neuron_splits[branch_idx]
    
    # Get node order (root → tip) and geodesic distances from root
    def get_branch_order_geodesic(sk, geodesic_mat_full):
        root = sk.nodes.loc[sk.nodes['type']=='root', 'node_id'].iat[0]
        node_order = [root]
        current = root
        while True:
            children = sk.nodes[sk.nodes['parent_id']==current]['node_id'].tolist()
            if not children:
                break
            node_order.extend(children)
            current = children[0]  # Take first child (could be improved)
        return node_order
    
    node_order = get_branch_order_geodesic(sk, geodesic_mat_full)
    root = node_order[0]
    dists_all = geodesic_mat_full.loc[root, node_order].values
    
    # Build density arrays along that order
    e_map = calculation_nodes.set_index('node_id')['synapse_density'].to_dict()
    e_all = np.array([e_map.get(n, 0.0) for n in node_order])
    
    # Drop the very first point at the root if it spikes
    dists = dists_all[1:]
    e_arr = e_all[1:]
    
    # Get every E-synapse's distance along this branch
    node_to_branch = {}
    for b, skn in enumerate(neuron_splits):
        for nid in skn.nodes['node_id']:
            node_to_branch[nid] = b
    
    df_e_branch = syn_exec_df[syn_exec_df['closest_node_id'].map(node_to_branch) == branch_idx]
    syn_dists = []
    
    # For each synapse, find its index in node_order and take the matching distance
    for nid in df_e_branch['closest_node_id']:
        idx = node_order.index(int(nid))
        if idx > 0:
            syn_dists.append(dists[idx - 1])
    syn_dists = np.array(syn_dists)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Draw E density curve
    ax.plot(dists, e_arr, '-', color='C0', lw=1.5, label='E density')
    
    # Draw vertical lines at each excitatory synapse location
    for x in syn_dists:
        j = np.argmin(np.abs(dists - x))
        ax.vlines(x, 0, e_arr[j], color='C0', alpha=.9, linewidth=.6)
    
    ax.set_xlabel("Distance from branch root (µm)")
    ax.set_ylabel("Synapse density")
    ax.set_title(f"Branch {branch_idx}: E density + E-synapse rug")
    ax.legend(loc='upper right', fontsize='small', frameon=False)
    plt.tight_layout()
    
    if save_plot:
        output_path = config.base_output_dir / "branches" / f"{config.neuron_id}_branch_{branch_idx}_density_analysis.{config.format}"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Branch density analysis plot saved to: {output_path}")
    
    return fig


def plot_branch_inhibitory_clusters(
    branch_idx: int,
    neuron_splits: List[Any],
    syn_exec_df: pd.DataFrame,
    syn_inh_df: pd.DataFrame,
    cluster_df_inh: pd.DataFrame,
    neuron_skel,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Plot 3D visualization of a branch with E/I synapses and I-cluster peaks.
    
    Args:
        branch_idx: Index of the branch to visualize
        neuron_splits: List of branch skeleton objects
        syn_exec_df: DataFrame with excitatory synapse data
        syn_inh_df: DataFrame with inhibitory synapse data
        cluster_df_inh: DataFrame with inhibitory cluster information
        neuron_skel: Neuron skeleton object
        config: Visualization configuration
        save_plot: Whether to save the plot
        
    Returns:
        Plotly figure
    """
    sk_branch = neuron_splits[branch_idx]
    
    # Build node→branch map
    node_to_branch = {}
    for b, sk in enumerate(neuron_splits):
        for nid in sk.nodes['node_id']:
            node_to_branch[nid] = b
    
    # Filter synapses to only those on this branch
    df_e = syn_exec_df[syn_exec_df['closest_node_id'].map(node_to_branch).eq(branch_idx)]
    df_i = syn_inh_df[syn_inh_df['closest_node_id'].map(node_to_branch).eq(branch_idx)]
    
    # Find I-cluster peaks on this branch
    i_peak_nodes = cluster_df_inh[cluster_df_inh['Cluster_Peak'].map(node_to_branch).eq(branch_idx)]['Cluster_Peak'].astype(int).tolist()
    peak_coords = sk_branch.nodes.set_index('node_id').loc[i_peak_nodes, ['x','y','z']]
    
    # Find the branch root coords
    root_row = sk_branch.nodes.loc[sk_branch.nodes['type']=='root'].iloc[0]
    root_xyz = (root_row['x'], root_row['y'], root_row['z'])
    
    # Build the figure
    fig = go.Figure()
    
    # Whole neuron faint grey (no legend)
    navis.plot3d(neuron_skel, fig=fig, color='lightgray', alpha=0.2, inline=False, legend=False)
    
    # This branch in green (no legend)
    navis.plot3d(sk_branch, fig=fig, color='green', linewidth=4, inline=False, legend=False)
    
    # Overlay E-synapses as blue circles
    fig.add_trace(go.Scatter3d(
        x=df_e['Epos3DX'],
        y=df_e['Epos3DY'],
        z=df_e['Epos3DZ'],
        mode='markers',
        marker=dict(size=5, color='blue', symbol='circle'),
        name='E-synapses'
    ))
    
    # Overlay I-synapses as red crosses
    fig.add_trace(go.Scatter3d(
        x=df_i['Ipos3DX'],
        y=df_i['Ipos3DY'],
        z=df_i['Ipos3DZ'],
        mode='markers',
        marker=dict(size=5, color='red', symbol='cross'),
        name='I-synapses'
    ))
    
    # Highlight I-cluster peaks as yellow diamonds
    if not peak_coords.empty:
        fig.add_trace(go.Scatter3d(
            x=peak_coords['x'],
            y=peak_coords['y'],
            z=peak_coords['z'],
            mode='markers',
            marker=dict(size=7, color='red', symbol='diamond'),
            name='I cluster peaks'
        ))
    
    # Mark the branch root as a larger green sphere
    fig.add_trace(go.Scatter3d(
        x=[root_xyz[0]],
        y=[root_xyz[1]],
        z=[root_xyz[2]],
        mode='markers',
        marker=dict(size=12, color='green', symbol='circle'),
        name='Branch root'
    ))
    
    # Zoom in to ±1 µm around this branch
    coords = sk_branch.nodes[['x','y','z']].to_numpy()
    mins, maxs = coords.min(axis=0) - 1, coords.max(axis=0) + 1
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[mins[0],maxs[0]], showticklabels=False),
            yaxis=dict(range=[mins[1],maxs[1]], showticklabels=False),
            zaxis=dict(range=[mins[2],maxs[2]], showticklabels=False),
            aspectmode='data'
        ),
        width=800,
        height=600,
        title=f"Branch {branch_idx}: E (blue), I (red), I peaks (yellow), root (lime)"
    )
    
    # Hide unwanted legend entries
    for trace in fig.data:
        if trace.name not in ('E-synapses','I-synapses','I cluster peaks','Branch root'):
            trace.showlegend = False
    
    if save_plot:
        output_path = config.base_output_dir / "branches" / f"{config.neuron_id}_branch_{branch_idx}_inhibitory_clusters.{config.format}"
        fig.write_image(str(output_path), width=800, height=600, scale=2)
        print(f"Branch inhibitory clusters plot saved to: {output_path}")
    
    return fig