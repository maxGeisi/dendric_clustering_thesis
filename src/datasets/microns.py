# src/data/microns.py
from pathlib import Path
import pandas as pd

def build_syn_path(syn_dir: str | Path, prefix: str, neuron_id: str) -> Path:
    syn_dir = Path(syn_dir)
    
    # Create directory if it doesn't exist
    syn_dir.mkdir(parents=True, exist_ok=True)
    
    p = syn_dir / f"{prefix}{neuron_id}.xlsx"
    if not p.exists():
        raise FileNotFoundError(f"Synapse file not found: {p}")
    return p

def read_synapses_raw(path: str | Path) -> pd.DataFrame:
    """Pure IO: read the MICrONS Excel and return the raw DataFrame (no transforms)."""
    return pd.read_excel(path, header=0)
