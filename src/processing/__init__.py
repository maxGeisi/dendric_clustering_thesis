# src/processing/__init__.py
from .synapse_mapping import (
    map_synapses_to_nodes,
    compute_synapse_node_statistics,
    create_node_counts_series
)

from .cluster_processing import (
    build_cluster_dataframe,
    compute_cluster_metrics,
    filter_clusters_by_density,
    mark_synapses_in_valid_clusters,
    add_inhibitory_synapse_counts,
    add_inhibitory_cluster_counts,
    print_cluster_statistics
)

from .inhibitory_analysis import (
    get_local_maximum_inh_e_gradient,
    create_inhibitory_clusters_by_e_gradient,
    map_inhibitory_to_excitatory_clusters_by_e_gradient,
    split_mixed_inhibitory_clusters_by_e_gradient,
    find_closest_excitatory_synapses_for_e_gradient,
    split_mixed_inhibitory_clusters,
    compute_e_i_relationships,
    compute_e_i_distance_analysis,
    print_inhibitory_cluster_statistics
)

from .distance_analysis import (
    find_closest_excitatory_synapses,
    map_inhibitory_to_excitatory_clusters_by_distance,
    split_mixed_inhibitory_clusters_by_distance,
    compute_distances_within_clusters,
    analyze_e_i_relationships,
    compute_distance_statistics,
    define_cutoff_strategies,
    apply_distance_cutoff,
    print_distance_analysis_summary
)

from .i_to_e_analysis import (
    run_complete_i_to_e_analysis,
    print_data_naming_conventions
)

from .branch_processing import (
    split_branches,
    build_branch_dataframe,
    filter_branches_by_length,
    add_volume_metrics,
    get_branches_with_both_synapse_types,
    get_branch_synapses,
    print_branch_filtering_summary
)

__all__ = [
    # synapse_mapping
    'map_synapses_to_nodes',
    'compute_synapse_node_statistics', 
    'create_node_counts_series',
    
    # cluster_processing
    'build_cluster_dataframe',
    'compute_cluster_metrics',
    'filter_clusters_by_density',
    'mark_synapses_in_valid_clusters',
    'add_inhibitory_synapse_counts',
    'add_inhibitory_cluster_counts',
    'print_cluster_statistics',
    
    # inhibitory_analysis
    'get_local_maximum_inh_e_gradient',
    'create_inhibitory_clusters_by_e_gradient',
    'map_inhibitory_to_excitatory_clusters_by_e_gradient',
    'split_mixed_inhibitory_clusters_by_e_gradient',
    'find_closest_excitatory_synapses_for_e_gradient',
    'split_mixed_inhibitory_clusters',
    'compute_e_i_relationships',
    'compute_e_i_distance_analysis',
    'print_inhibitory_cluster_statistics',
    
    # distance_analysis
    'find_closest_excitatory_synapses',
    'map_inhibitory_to_excitatory_clusters_by_distance',
    'split_mixed_inhibitory_clusters_by_distance',
    'compute_distances_within_clusters',
    'analyze_e_i_relationships',
    'compute_distance_statistics',
    'define_cutoff_strategies',
    'apply_distance_cutoff',
    'print_distance_analysis_summary',
    
    # i_to_e_analysis
    'run_complete_i_to_e_analysis',
    'print_data_naming_conventions',
    
    # branch_processing
    'split_branches',
    'build_branch_dataframe',
    'filter_branches_by_length',
    'add_volume_metrics',
    'get_branches_with_both_synapse_types',
    'get_branch_synapses',
    'print_branch_filtering_summary'
]
