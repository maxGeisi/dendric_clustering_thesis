# Statistical Analysis Visualizations

This module provides statistical comparison visualizations for cluster properties, specifically comparing excitatory clusters with and without associated inhibitory synapses.

## Features

The statistical analysis module creates violin plots with statistical significance testing for three key cluster properties:

1. **Minimal Cable Length** - The minimum spanning tree length of cluster nodes
2. **E-Synapse Count** - Number of excitatory synapses per cluster
3. **Cluster Density** - Synapses per unit cable length (synapses/μm)

## Statistical Testing

- **Test**: Mann-Whitney U test (non-parametric)
- **Significance levels**:
  - `***` p ≤ 0.001 (highly significant)
  - `**` p ≤ 0.01 (very significant)  
  - `*` p ≤ 0.05 (significant)
  - `ns` p > 0.05 (not significant)

## Required Data

The `cluster_df` DataFrame must contain these columns:

- `minimal_cable_length`: Float values in micrometers
- `e_synapse_count`: Integer counts of excitatory synapses
- `cluster_density`: Float values (synapses/μm)
- `has_I_associated`: Boolean indicating if cluster has associated inhibitory synapses

## Usage

### Basic Usage

```python
from src.visualization.visualization_orchestrator import create_statistical_comparison_visualizations

# Create all statistical comparison plots
figures = create_statistical_comparison_visualizations(
    cluster_df=cluster_df,
    neuron_id="n3",
    output_base_dir=Path("figures/microns/n3"),
    save_plots=True
)
```

### Individual Plot Functions

```python
from src.visualization.statistical_plots import (
    plot_cluster_cable_length_comparison,
    plot_cluster_synapse_count_comparison,
    plot_cluster_density_comparison
)

# Create individual plots
cable_fig = plot_cluster_cable_length_comparison(cluster_df, config, save_plot=True)
count_fig = plot_cluster_synapse_count_comparison(cluster_df, config, save_plot=True)
density_fig = plot_cluster_density_comparison(cluster_df, config, save_plot=True)
```

## Output Files

Plots are saved to the `analysis/` subdirectory with standardized naming:

- `{neuron_id}_cable_length_i_comparison.png`
- `{neuron_id}_synapse_count_i_comparison.png`
- `{neuron_id}_cluster_density_i_comparison.png`

## Plot Features

Each violin plot includes:

- **Violin plots** with box plots inside showing distribution shapes
- **Statistical significance bracket** connecting the two groups
- **Significance star** above the bracket indicating p-value level
- **Mean and median annotations** below each violin
- **Standardized colors** for consistent visualization

## Example Output

The plots will show:
- Left violin: Clusters without I synapses ("No I")
- Right violin: Clusters with I synapses ("With I")
- Statistical comparison with appropriate significance level
- Summary statistics (mean and median) for each group

## Integration

This module integrates with the main visualization orchestrator and can be used alongside other visualization functions in the analysis pipeline.
