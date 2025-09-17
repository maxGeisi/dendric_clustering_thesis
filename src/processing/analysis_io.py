# src/processing/analysis_io.py
"""
Simple directory creation for analysis results.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict


def create_analysis_directories(base_dir: Path) -> Dict[str, Path]:
    """
    Create organized directory structure for analysis results.
    
    Args:
        base_dir: Base directory for analysis results
        
    Returns:
        Dictionary with directory paths
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different analysis types
    directories = {
        'analysis_root': base_dir,
        'mapping': base_dir / "i_to_e_mapping",
        'distance': base_dir / "distance_analysis",
        'relationships': base_dir / "e_i_relationships",
        'filtered_data': base_dir / "filtered_data"
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories
