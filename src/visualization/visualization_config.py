# src/visualization/visualization_config.py
"""
Configuration and utilities for visualization output management.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """Configuration for visualization output."""
    base_output_dir: Path
    neuron_id: str
    figure_width: int = 1200
    figure_height: int = 800
    dpi: int = 300
    format: str = "png"
    
    def __post_init__(self):
        """Create output directories after initialization."""
        self.create_directories()
    
    def create_directories(self) -> None:
        """Create all necessary output directories."""
        directories = [
            self.base_output_dir,
            self.base_output_dir / "synapses",
            self.base_output_dir / "nodes", 
            self.base_output_dir / "density",
            self.base_output_dir / "clusters",
            self.base_output_dir / "analysis",
            self.base_output_dir / "t_test_violin",
            self.base_output_dir / "histo_plots"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_output_path(self, category: str, filename: str) -> Path:
        """Get full output path for a visualization file."""
        return self.base_output_dir / category / filename
    
    def get_filename(self, plot_type: str, suffix: str = "") -> str:
        """Generate standardized filename for plots."""
        base_name = f"{self.neuron_id}_{plot_type}"
        if suffix:
            base_name += f"_{suffix}"
        return f"{base_name}.{self.format}"


def create_visualization_config(
    base_output_dir: Path,
    neuron_id: str,
    **kwargs
) -> VisualizationConfig:
    """
    Create visualization configuration with standardized naming.
    
    Args:
        base_output_dir: Base directory for all visualizations
        neuron_id: Neuron identifier (e.g., "n3")
        **kwargs: Additional configuration parameters
        
    Returns:
        VisualizationConfig object
    """
    return VisualizationConfig(
        base_output_dir=base_output_dir,
        neuron_id=neuron_id,
        **kwargs
    )


def get_standard_colors() -> Dict[str, str]:
    """Get standardized color scheme for visualizations."""
    return {
        "excitatory": "#1f77b4",  # Blue
        "inhibitory": "#d62728",  # Red
        "skeleton": "#2ca02c",    # Green
        "skeleton_grey": "#808080", # Grey
        "peak_node": "#ff7f0e",   # Orange
        "high_density": "#ff0000", # Red
        "low_density": "#000000",  # Black
        "medium_density": "#ffff00" # Yellow
    }


def get_standard_markers() -> Dict[str, Dict[str, Any]]:
    """Get standardized marker configurations."""
    return {
        "excitatory": {"size": 2, "color": "#1f77b4"},
        "inhibitory": {"size": 2, "color": "#d62728"},
        "nodes_with_synapses": {"size": 3, "color": "#1f77b4"},
        "peak_node": {"size": 6, "color": "#ff7f0e"},
        "max_synapse_node": {"size": 6, "color": "#d62728"}
    }
