#!/usr/bin/env python3
"""
Setup script to ensure all data directories are created automatically.
This can be run as a standalone script or imported into the main notebook.
"""

from pathlib import Path
from .utils import ensure_data_directories
from .config import load_config

def setup_all_directories(config_path: str = "config/default.yaml") -> dict:
    """
    Set up all necessary directories for the analysis.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with all created directory paths
    """
    # Load configuration
    cfg = load_config(config_path)
    
    # Ensure data directories exist
    data_dirs = ensure_data_directories(cfg.paths.base_data, cfg.dataset_id)
    
    print("=" * 60)
    print("SETTING UP DATA DIRECTORIES")
    print("=" * 60)
    print(f"Dataset: {cfg.dataset_id}")
    print(f"Neuron ID: {cfg.input.neuron_id}")
    print()
    
    print("Created/verified directories:")
    for name, path in data_dirs.items():
        print(f"  ✅ {name}: {path}")
    
    print()
    print("Next steps:")
    print(f"  1. Place SWC files in: {data_dirs['swc_skel']}")
    print(f"  2. Place synapse files in: {data_dirs['syn']}")
    print(f"  3. Run the analysis - output will go to: figures/{cfg.dataset_id}/{cfg.input.neuron_id}/")
    print(f"  4. Cache files will be stored in: {data_dirs['interim_results']}")
    
    return data_dirs

if __name__ == "__main__":
    setup_all_directories()
