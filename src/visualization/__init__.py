# src/visualization/__init__.py
"""
Visualization module for dendric clustering analysis.
"""

# Import the new visualization system
from .visualization_config import VisualizationConfig, create_visualization_config
from .synapse_plots import plot_3d_synapses, plot_excitatory_synapses_only
from .node_plots import plot_node_synapse_counts, plot_node_synapse_counts_simple
from .density_plots import plot_synapse_density, plot_synapse_density_percentage_based
from .cluster_visualization import plot_clusters_by_synapse_count, plot_clusters_by_density
from .visualization_orchestrator import create_all_visualizations

# Import existing cluster and histogram plots if they exist
try:
    from .cluster_plots import (
        plot_clusters_3d,
        plot_clusters_by_size,
        get_cluster_color_scheme
    )
    _has_cluster_plots = True
except ImportError:
    _has_cluster_plots = False

try:
    from .histogram_plots import (
        plot_cluster_size_histogram,
        plot_cluster_density_histogram,
        plot_e_i_distance_histogram
    )
    _has_histogram_plots = True
except ImportError:
    _has_histogram_plots = False

# Define __all__ based on what's available
__all__ = [
    # New visualization system
    'VisualizationConfig',
    'create_visualization_config',
    'plot_3d_synapses',
    'plot_excitatory_synapses_only',
    'plot_node_synapse_counts',
    'plot_node_synapse_counts_simple',
    'plot_synapse_density',
    'plot_synapse_density_percentage_based',
    'plot_clusters_by_synapse_count',
    'plot_clusters_by_density',
    'create_all_visualizations'
]

# Add existing plots if available
if _has_cluster_plots:
    __all__.extend([
        'plot_clusters_3d',
        'plot_clusters_by_size',
        'get_cluster_color_scheme'
    ])

if _has_histogram_plots:
    __all__.extend([
        'plot_cluster_size_histogram',
        'plot_cluster_density_histogram',
        'plot_e_i_distance_histogram'
    ])
