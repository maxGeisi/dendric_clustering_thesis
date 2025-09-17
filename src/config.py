from pathlib import Path
import yaml
from dataclasses import dataclass

@dataclass
class VoxelNM:
    dx: float
    dy: float
    dz: float

@dataclass
class Paths:
    base_data: str
    swc_dir: str
    syn_dir: str
    interim_results_dir: str   

@dataclass
class InputSpec:
    neuron_id: str
    swc_prefix: str
    syn_prefix: str

@dataclass
class Analysis:
    sigma_um: float
    distance_mapping: bool

@dataclass
class Output:
    figures_dir: str

@dataclass
class Config:
    dataset_id: str
    paths: Paths
    input: InputSpec
    voxel_nm: VoxelNM
    analysis: Analysis
    output: Output

def load_config(path: str | Path = "config/default.yaml") -> Config:
    with open(path, "r") as f:
        y = yaml.safe_load(f)

    return Config(
        dataset_id=y["dataset_id"],
        paths=Paths(**y["paths"]),
        input=InputSpec(**y["input"]),
        voxel_nm=VoxelNM(**y["voxel_nm"]),
        analysis=Analysis(**y["analysis"]),
        output=Output(**y["output"]),
    )
