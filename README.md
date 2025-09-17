# Dendric Clustering Analysis

A comprehensive Python package for analyzing dendritic clustering of excitatory and inhibitory synapses in neuronal morphology data. This project provides tools for synapse mapping, clustering analysis, statistical comparisons, and advanced visualizations.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Input Data Requirements](#input-data-requirements)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Performance Notes](#performance-notes)
- [Manual Configuration](#manual-configuration)
- [Still in Progress](#still-in-progress)

## Overview

This project performs comprehensive dendritic clustering analysis including:

- **Synapse Mapping**: Maps excitatory and inhibitory synapses to neuronal skeleton nodes
- **Density Analysis**: Computes synapse density across dendritic branches
- **Clustering**: Identifies clusters of excitatory synapses using density-based methods
- **Statistical Analysis**: Compares cluster properties between groups with/without inhibitory synapses
- **Visualization**: Creates comprehensive 3D and statistical visualizations
- **Branch Analysis**: Analyzes synapse distribution across dendritic branches

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda package manager
- Git (for cloning the repository)

### Quick Start for New Users

When you clone this repository, you'll get:
- ✅ Source code and configuration files
- ✅ Directory structure with `.gitkeep` files
- ❌ **No data files** (you need to add your own)
- ❌ **No conda environment** (you need to create your own)
- ❌ **No generated figures** (they'll be created when you run analyses)

### Setup Conda Environment

**Important**: The conda environment is created on your local system, not in the repository. When users clone your repository, they need to set up their own environment.

#### Option 1: Quick Setup (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd dendric_clustering/refactored
   ```

2. **Run the setup script:**
   ```bash
   python setup.py
   ```
   This will show you the exact commands to run.

#### Option 2: Manual Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd dendric_clustering/refactored
   ```

2. **Create a new conda environment:**
   ```bash
   conda create -n dendric python=3.9
   conda activate dendric
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import navis, pandas, numpy, plotly; print('Installation successful!')"
   ```

**Note**: Each user needs to create their own conda environment locally. The environment is not part of the repository and cannot be shared directly.

### Dependencies

The project requires the following packages (see `requirements.txt` for versions):

- **Core**: numpy, pandas, scipy
- **Neuroinformatics**: navis
- **Visualization**: matplotlib, plotly, seaborn
- **Data Processing**: openpyxl, pyyaml
- **Analysis**: scikit-learn, statsmodels
- **Utilities**: tqdm, jupyter, ipykernel

## Project Structure

```
refactored/
├── config/
│   └── default.yaml          # Configuration file
├── data/
│   ├── cache/                # Cached computation results (auto-created)
│   ├── microns/              # MICrONS dataset (place your data here)
│   │   ├── extracted_swc/
│   │   │   ├── SWC_Skel/     # SWC skeleton files
│   │   │   └── syn/          # Synapse data files
│   │   └── intirim_results/  # Intermediate results (auto-created)
│   └── iascone/              # IASCONE dataset (place your data here)
├── figures/                  # Generated visualizations (auto-created)
│   └── {dataset}/{neuron_id}/ # Neuron-specific output directories
├── main/
│   └── thesis_script_main.ipynb  # Main analysis notebook
├── src/                      # Source code
│   ├── analysis/             # Analysis modules
│   ├── datasets/             # Dataset loaders
│   ├── processing/           # Data processing
│   ├── visualization/        # Visualization modules
│   └── *.py                  # Core modules
├── .gitignore               # Git ignore rules
├── setup.py                 # Setup helper script
└── requirements.txt          # Python dependencies
```

**Note**: The `data/` and `figures/` directories are tracked by git but their contents are ignored. Place your data files in the appropriate subdirectories, and the code will automatically create the necessary output directories when running analyses.

## Input Data Requirements

### File Naming Convention

#### SWC Skeleton Files
Place SWC files in `data/{dataset}/extracted_swc/SWC_Skel/` with the naming convention:

ALL OF THESE CAN ALSO BE CHANGED IN THE default.yaml FILE, TO FIT YOUR NEEDS!

**MICrONS dataset:**
- Format: `{prefix}{neuron_id}.swc`
- Example: `microns_jan_n3.swc`
- Where `prefix` is defined in config (default: `microns_jan_`)

**IASCONE dataset:**
- Format: `{neuron_id}.swc`
- Example: `14.swc`, `18.swc`, etc.

#### Synapse Data Files
Place synapse files in `data/{dataset}/extracted_swc/syn/` with the naming convention:

**MICrONS dataset:**
- Format: `{prefix}syn{neuron_id}.xlsx`
- Example: `microns_jan_synn3.xlsx`

**IASCONE dataset:**
- Format: `{neuron_id}_exc_syn.csv` and `{neuron_id}_inh_syn.csv`
- Example: `14_exc_syn.csv`, `14_inh_syn.csv`

### SWC File Structure

SWC files must follow the standard SWC format with these columns:

```
# PointNo Label X Y Z Radius Parent
1 0 651.01 231.87 906.80 0.79 -1
2 0 662.09 220.09 895.14 1.01 -1
...
```

**Column descriptions:**
- `PointNo`: Unique node identifier
- `Label`: Node type (0=undefined, 1=soma, 5=fork, 6=end)
- `X, Y, Z`: 3D coordinates in micrometers
- `Radius`: Node radius in micrometers
- `Parent`: Parent node ID (-1 for root)

### Synapse Data Structure

Synapse data files must contain these columns:

**Required columns:**
- `x, y, z`: 3D coordinates in nanometers
- `syn_size`: Synapse size information
- `is_on_spine`: Boolean indicating spine location (True=inhibitory, False=excitatory)

**Optional columns:**
- Additional metadata columns are preserved

## Configuration

The analysis is controlled through the YAML configuration file (`config/default.yaml`):

```yaml
dataset_id: "microns"                # "microns" or "iascone"

paths:
  base_data: "data"
  swc_dir: "data/microns/extracted_swc/SWC_Skel"
  syn_dir: "data/microns/extracted_swc/syn"
  interim_results_dir: "data/microns/intirim_results"

input:
  neuron_id: "n3"                    # Target neuron ID
  swc_prefix: "microns_jan_"         # SWC file prefix
  syn_prefix: "microns_jan_syn"      # Synapse file prefix

voxel_nm:
  dx: 10                             # X voxel size in nm
  dy: 10                             # Y voxel size in nm
  dz: 40                             # Z voxel size in nm

analysis:
  sigma_um: 1.0                      # Gaussian kernel sigma
  distance_mapping: true             # Enable distance mapping

output:
  figures_dir: "figures"             # Output directory
```

### Configuration Parameters

- **`neuron_id`**: The specific neuron to analyze (e.g., "n3", "14")
- **`swc_prefix`/`syn_prefix`**: File naming prefixes for the dataset
- **`voxel_nm`**: Voxel dimensions for volume calculations
- **`sigma_um`**: Gaussian kernel parameter for density calculations
- **`distance_mapping`**: Whether to perform inhibitory-to-excitatory distance mapping

## Usage

### Running the Analysis

1. **Activate the environment:**
   ```bash
   conda activate dendric
   ```

2. **Navigate to the project directory:**
   ```bash
   cd dendric_clustering/refactored
   ```

3. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

4. **Open the main notebook:**
   ```
   main/thesis_script_main.ipynb
   ```

5. **Configure the analysis:**
   - Edit `config/default.yaml` to set your target neuron and parameters
   - Place your input files in the appropriate `data/{dataset}/extracted_swc/` directories

6. **Run the analysis:**
   - Execute cells sequentially from top to bottom
   - Each cell performs a specific analysis step
   - Results are automatically saved to the output directories

### Environment Management

**For new users:**
- Each user must create their own conda environment locally
- The environment name (`dendric`) is just a suggestion - users can choose any name
- The environment is not part of the repository

**For development:**
- Always activate your environment before working: `conda activate dendric`
- Update `requirements.txt` if you add new dependencies
- Test your environment setup with the verification command

### Analysis Workflow

The main notebook follows this workflow:

1. **Configuration & Setup** (Cells 1-3)
2. **Neuron Skeleton Loading** (Cell 4)
3. **Synapse Data Loading** (Cell 5)
4. **Synapse-to-Node Mapping** (Cell 6)
5. **Geodesic Distance Matrix** (Cell 7)
6. **Node Weight Matrix** (Cell 8)
7. **Density Analysis** (Cell 9)
8. **Clustering Analysis** (Cell 10)
9. **Cluster Processing** (Cell 11)
10. **Distance Analysis** (Cell 12)
11. **Visualization** (Cells 13-25)
12. **Advanced Branch Analysis** (Cell 26+)

## Output Structure

### Generated Figures

Figures are saved to `figures/{dataset}/{neuron_id}/` with the following structure:

```
figures/microns/n3/
├── synapses/                    # 3D synapse visualizations
├── nodes/                      # Node-based visualizations
├── density/                    # Density analysis plots
├── clusters/                   # Cluster visualizations
├── analysis/                   # Statistical analysis
├── t_test_violin/             # Statistical comparisons
├── histo_plots/               # Histogram analyses
└── branches/                  # Branch-specific analyses
```

### Cached Results

Computed matrices are cached in `data/{dataset}/intirim_results/`:

- `geodesic_mat_{neuron_id}.pkl`: Geodesic distance matrix
- `node_weight_mat_{neuron_id}_sigma{sigma}.pkl`: Node weight matrix
- `exec_syn_distance_mat_{neuron_id}.pkl`: Synapse distance matrix

## Performance Notes

### Computation Time

- **First run**: Initial computation of geodesic and weight matrices can take 10-30 minutes for large neurons (30k+ nodes)
- **Subsequent runs**: Cached matrices are loaded instantly
- **Memory usage**: Large matrices require 4-8 GB RAM for typical neurons

### Caching System

The system automatically caches expensive computations:

- **Geodesic matrices**: Computed once per neuron, reused across analyses
- **Weight matrices**: Cached per neuron and sigma parameter
- **Distance matrices**: Cached per analysis run

To force recomputation, set `force_recompute=True` in the relevant cells.

## Manual Configuration

Some analysis steps require manual parameter adjustment:

### Advanced Branch Visualization (Cell 26)

For branch-specific analyses, you need to manually specify:

```python
# Example: Analyze branch 51
branch_idx = 51  # Change this to your target branch

# Run advanced branch visualizations
results = create_advanced_branch_visualizations(
    branch_idx=branch_idx,
    # ... other parameters
)
```

### Statistical Analysis Parameters

Adjust statistical test parameters in the visualization cells:

```python
# Example: Change significance levels
def get_significance_star(p):
    if p <= 0.001: return '***'
    elif p <= 0.01:  return '**'
    elif p <= 0.05:  return '*'
    else:            return 'ns'
```

### Visualization Parameters

Customize plot appearance in the config:

```python
# Example: Change figure dimensions
config = create_visualization_config(
    base_output_dir=output_dir,
    neuron_id=neuron_id,
    figure_width=1600,  # Increase width
    figure_height=1000, # Increase height
    dpi=300            # High resolution
)
```

## Still in Progress

The following features are currently under development:

- **IASCONE dataset**: Integration of additional dataset format
- **Path/off-path inhibition**: Analysis of inhibitory synapse positioning relative to excitatory pathways
- **Many iterations for control group**: Statistical validation through multiple iterations and control group comparisons


## Git Repository Setup

### .gitignore Configuration

The project includes a comprehensive `.gitignore` file that ensures:

- **Data files are not uploaded**: All files in `data/` directories are ignored
- **Generated figures are not uploaded**: All files in `figures/` directories are ignored  
- **Cache files are not uploaded**: All `.pkl`, `.pickle`, and cache files are ignored
- **System files are ignored**: VS Code, macOS, Windows, and Linux system files
- **Python artifacts are ignored**: `__pycache__/`, `.pyc` files, virtual environments
- **Directory structure is preserved**: `.gitkeep` files maintain empty directories

### Directory Structure

The code automatically creates the necessary directory structure:

- **Figures directories**: Created by `VisualizationConfig` class when running analyses
- **Cache directories**: Created when computing geodesic and weight matrices
- **Analysis directories**: Created when saving analysis results

You only need to place your input data files in the appropriate `data/{dataset}/extracted_swc/` directories.

## Contributing

This project is part of ongoing research in dendritic clustering analysis. For questions or contributions, please refer to the project documentation or contact the development team.

## License

This project is developed for research purposes. Please cite appropriately if used in academic work.
