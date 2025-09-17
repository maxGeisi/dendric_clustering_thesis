# src/visualization/synapse_plots.py
"""
3D synapse visualization functions.
"""
from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
import navis
import pandas as pd
import scipy.stats as stats
from pathlib import Path
from typing import Optional

from .visualization_config import VisualizationConfig, get_standard_colors, get_standard_markers


def plot_3d_synapses(
    syn_exec_df: pd.DataFrame,
    syn_inh_df: pd.DataFrame,
    neuron_skel,
    config: VisualizationConfig,
    show_inhibitory: bool = True,
    save_plot: bool = True
) -> go.Figure:
    """
    Create 3D visualization of excitatory and inhibitory synapses with neuron skeleton.
    
    Args:
        syn_exec_df: DataFrame with excitatory synapses (Epos3DX, Epos3DY, Epos3DZ)
        syn_inh_df: DataFrame with inhibitory synapses (Ipos3DX, Ipos3DY, Ipos3DZ)
        neuron_skel: Navis neuron skeleton object
        config: Visualization configuration
        show_inhibitory: Whether to show inhibitory synapses
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    colors = get_standard_colors()
    markers = get_standard_markers()
    
    # Check for required columns
    required_e_cols = ['Epos3DX', 'Epos3DY', 'Epos3DZ']
    required_i_cols = ['Ipos3DX', 'Ipos3DY', 'Ipos3DZ']
    
    missing_e_cols = [col for col in required_e_cols if col not in syn_exec_df.columns]
    missing_i_cols = [col for col in required_i_cols if col not in syn_inh_df.columns]
    
    if missing_e_cols:
        print(f"Warning: Missing excitatory synapse columns: {missing_e_cols}. Skipping 3D synapse visualization.")
        fig = go.Figure()
        fig.update_layout(
            title=f"3D Synapse Visualization - {config.neuron_id} (Data Not Available)",
            annotations=[dict(text="Required excitatory synapse columns not available", x=0.5, y=0.5, showarrow=False)]
        )
        return fig
    
    # Initialize 3D figure
    fig = go.Figure(layout=dict(
        width=config.figure_width,
        height=config.figure_height,
        title=f"3D Synapse Visualization - {config.neuron_id}",
        scene=dict(
            xaxis_title="X (μm)",
            yaxis_title="Y (μm)", 
            zaxis_title="Z (μm)",
            aspectmode="data"
        )
    ))
    
    # Plot excitatory synapses in blue
    fig.add_trace(go.Scatter3d(
        x=syn_exec_df['Epos3DX'],
        y=syn_exec_df['Epos3DY'],
        z=syn_exec_df['Epos3DZ'],
        mode='markers',
        marker=dict(
            size=markers["excitatory"]["size"],
            color=markers["excitatory"]["color"],
            opacity=0.8
        ),
        name='Excitatory synapses'
    ))
    
    # Plot inhibitory synapses in red (if requested and columns available)
    if show_inhibitory and len(syn_inh_df) > 0 and not missing_i_cols:
        fig.add_trace(go.Scatter3d(
            x=syn_inh_df['Ipos3DX'],
            y=syn_inh_df['Ipos3DY'],
            z=syn_inh_df['Ipos3DZ'],
            mode='markers',
            marker=dict(
                size=markers["inhibitory"]["size"],
                color=markers["inhibitory"]["color"],
                opacity=0.8
            ),
            name='Inhibitory synapses'
        ))
    elif show_inhibitory and missing_i_cols:
        print(f"Warning: Missing inhibitory synapse columns: {missing_i_cols}. Showing only excitatory synapses.")
    
    # Overlay neuron skeleton
    navis.plot3d(
        neuron_skel,
        fig=fig,
        color="green",
        palette="viridis",
        legend=True,
        inline=False
    )
    
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
    
    # Save plot if requested
    if save_plot:
        suffix = "both" if show_inhibitory else "excitatory_only"
        filename = config.get_filename("synapses_3d", suffix)
        output_path = config.get_output_path("synapses", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved synapse visualization: {output_path}")
    
    return fig


def plot_excitatory_synapses_only(
    syn_exec_df: pd.DataFrame,
    neuron_skel,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create 3D visualization of excitatory synapses only with neuron skeleton.
    
    Args:
        syn_exec_df: DataFrame with excitatory synapses (Epos3DX, Epos3DY, Epos3DZ)
        neuron_skel: Navis neuron skeleton object
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    colors = get_standard_colors()
    markers = get_standard_markers()
    
    # Check for required columns
    required_e_cols = ['Epos3DX', 'Epos3DY', 'Epos3DZ']
    missing_e_cols = [col for col in required_e_cols if col not in syn_exec_df.columns]
    
    if missing_e_cols:
        print(f"Warning: Missing excitatory synapse columns: {missing_e_cols}. Skipping excitatory synapse visualization.")
        fig = go.Figure()
        fig.update_layout(
            title=f"Excitatory Synapses - {config.neuron_id} (Data Not Available)",
            annotations=[dict(text="Required excitatory synapse columns not available", x=0.5, y=0.5, showarrow=False)]
        )
        return fig
    
    # Initialize 3D figure
    fig = go.Figure(layout=dict(
        width=config.figure_width,
        height=config.figure_height,
        title=f"Excitatory Synapses - {config.neuron_id}",
        scene=dict(
            xaxis_title="X (μm)",
            yaxis_title="Y (μm)",
            zaxis_title="Z (μm)",
            aspectmode="data"
        )
    ))
    
    # Plot excitatory synapses with larger markers
    fig.add_trace(go.Scatter3d(
        x=syn_exec_df['Epos3DX'],
        y=syn_exec_df['Epos3DY'],
        z=syn_exec_df['Epos3DZ'],
        mode='markers',
        marker=dict(
            size=4,  # Larger than default
            color=markers["excitatory"]["color"],
            opacity=0.8
        ),
        name='Excitatory synapses'
    ))
    
    # Overlay neuron skeleton in grey
    navis.plot3d(
        neuron_skel,
        fig=fig,
        color=colors["skeleton_grey"],
        palette="viridis",
        legend=False,
        inline=False
    )
    
    # Keep only the synapse legend entries; hide any colorbars and neuron IDs
    keep = {"Excitatory synapses"}
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
        filename = config.get_filename("synapses_3d", "excitatory_only")
        output_path = config.get_output_path("synapses", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved excitatory synapse visualization: {output_path}")
    
    return fig


def plot_distance_to_soma(
    syn_exec_df: pd.DataFrame,
    syn_inh_df: pd.DataFrame,
    neuron_skel,
    config: VisualizationConfig,
    save_plot: bool = True
) -> go.Figure:
    """
    Create violin plot comparing distance to soma between excitatory and inhibitory synapses.
    
    Args:
        syn_exec_df: DataFrame with excitatory synapse data including distance_to_soma
        syn_inh_df: DataFrame with inhibitory synapse data including distance_to_soma
        neuron_skel: Neuron skeleton object (unused but kept for API compatibility)
        config: Visualization configuration
        save_plot: Whether to save the plot to file
        
    Returns:
        Plotly figure object
    """
    # Check if distance_to_soma column exists, if not, skip this visualization
    if "distance_to_soma" not in syn_exec_df.columns or "distance_to_soma" not in syn_inh_df.columns:
        print("Warning: distance_to_soma column not found in synapse DataFrames. Skipping distance to soma visualization.")
        # Return an empty figure
        fig = go.Figure()
        fig.update_layout(
            title=f"Distance to Soma Visualization - {config.neuron_id} (Data Not Available)",
            annotations=[dict(text="distance_to_soma column not available", x=0.5, y=0.5, showarrow=False)]
        )
        return fig
    
    # 1) Gather the two series and tag them
    exec_dist = syn_exec_df[["distance_to_soma"]].copy()
    exec_dist["Type"] = "E-synapse"

    inh_dist = syn_inh_df[["distance_to_soma"]].copy()
    inh_dist["Type"] = "I-synapse"

    # 2) Concatenate into one DataFrame
    df_dist = pd.concat([exec_dist, inh_dist], ignore_index=True)

    # 3) Compute p-value (Welch's t-test) and convert to star
    e_vals = exec_dist["distance_to_soma"].dropna()
    i_vals = inh_dist["distance_to_soma"].dropna()
    pval = stats.ttest_ind(e_vals, i_vals, equal_var=False).pvalue

    def get_significance_star(p):
        if p <= 0.001: return '***'
        elif p <= 0.01:  return '**'
        elif p <= 0.05:  return '*'
        else:            return 'ns'

    star = get_significance_star(pval)

    # 4) Build the violin plot
    fig = px.violin(
        df_dist,
        x="Type",
        y="distance_to_soma",
        box=True,
        points=False,
        title=f"Distance to Soma by Synapse Type - {config.neuron_id}",
        labels={"distance_to_soma": "Distance to Soma (μm)", "Type": ""}
    )

    # 5) Add bracket + star between the two violins
    ymax = df_dist["distance_to_soma"].max()
    bracket_y = ymax * 1.15  # 15% above the tallest violin to ensure it's clearly above

    fig.add_shape(
        type="line",
        x0=0, x1=1,
        y0=bracket_y, y1=bracket_y,
        xref="x", yref="y",
        line=dict(color="black", width=1.5)
    )
    fig.add_annotation(
        x=0.5, y=bracket_y,
        text=star,
        showarrow=False,
        xref="x", yref="y",
        font=dict(size=16),
        yshift=5  # Reduced yshift since line is already higher
    )

    fig.update_layout(margin=dict(t=60))

    # Print summary statistics
    print("Distance to soma mean E synapse: ", syn_exec_df["distance_to_soma"].mean())
    print("Distance to soma mean I synapse: ", syn_inh_df["distance_to_soma"].mean())

    # Save plot if requested
    if save_plot:
        filename = config.get_filename("distance_to_soma_syn_type", "")
        output_path = config.get_output_path("synapses", filename)
        fig.write_image(str(output_path), width=config.figure_width, height=config.figure_height, scale=2)
        # Also save as SVG
        svg_path = output_path.with_suffix('.svg')
        fig.write_image(str(svg_path), width=config.figure_width, height=config.figure_height, scale=2)
        print(f"Saved distance to soma visualization: {output_path}")

    return fig