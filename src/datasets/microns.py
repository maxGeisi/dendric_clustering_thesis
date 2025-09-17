# src/data/microns.py
from pathlib import Path
import pandas as pd

def build_syn_path(syn_dir: str | Path, prefix: str, neuron_id: str) -> Path:
    p = Path(syn_dir) / f"{prefix}{neuron_id}.xlsx"
    if not p.exists():
        raise FileNotFoundError(f"Synapse file not found: {p}")
    return p

def read_synapses_raw(path: str | Path) -> pd.DataFrame:
    """Pure IO: read the MICrONS Excel and return the raw DataFrame (no transforms)."""
    return pd.read_excel(path, header=0)
