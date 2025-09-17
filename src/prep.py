# src/prep.py
from __future__ import annotations
import pandas as pd

# ---------- RAW-layer ops (work on original columns) ----------

def dedupe_raw_xyz(df: pd.DataFrame, x="x", y="y", z="z") -> pd.DataFrame:
    """Exact duplicate removal by raw x,y,z columns (original behavior)."""
    return df.drop_duplicates(subset=[x, y, z], keep="first").reset_index(drop=True)

def add_synapse_volume_nm3_to_um3(df: pd.DataFrame,
                                  syn_size_col: str = "syn_size",
                                  out_col: str = "SynapseVolume",
                                  dx_nm: float = 10, dy_nm: float = 10, dz_nm: float = 40) -> pd.DataFrame:
    """Compute volume in µm³ from syn_size * (dx*dy*dz) nm³."""
    df = df.copy()
    if syn_size_col in df.columns:
        voxel_vol_um3 = (dx_nm * dy_nm * dz_nm) / 1e9
        df[out_col] = df[syn_size_col] * voxel_vol_um3
    return df

def split_e_i_by_isOnSpine(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Original rule: isOnSpine==0 → E (spine), isOnSpine==1 → I (shaft).
    Returns (syn_exec_df with Epos3D*, syn_inh_df with Ipos3D*).
    """
    if "isOnSpine" not in df.columns:
        raise KeyError("Expected 'isOnSpine' column in synapse table.")

    # Legacy coordinate names for E branch
    dfe = df.rename(columns={"x": "Epos3DX", "y": "Epos3DY", "z": "Epos3DZ"})
    syn_exec_df = dfe.loc[dfe["isOnSpine"] == 0].copy()
    syn_inh_df  = dfe.loc[dfe["isOnSpine"] == 1].copy()
    # For inhibitory, rename to Ipos3D*
    syn_inh_df = syn_inh_df.rename(columns={"Epos3DX": "Ipos3DX",
                                            "Epos3DY": "Ipos3DY",
                                            "Epos3DZ": "Ipos3DZ"})
    return syn_exec_df, syn_inh_df

# ---------- NORMALIZED-layer ops (standard schema for new pipeline) ----------

def normalize_position_columns(df: pd.DataFrame, x="x", y="y", z="z", unit="nm") -> pd.DataFrame:
    """Create x_um,y_um,z_um in micrometers from raw x,y,z."""
    df = df.copy()
    if unit.lower() == "nm":
        df["x_um"] = df[x] / 1000.0
        df["y_um"] = df[y] / 1000.0
        df["z_um"] = df[z] / 1000.0
    elif unit.lower() == "um":
        df["x_um"], df["y_um"], df["z_um"] = df[x], df[y], df[z]
    else:
        raise ValueError(f"Unknown unit: {unit}")
    return df

def compute_volume_um3(df: pd.DataFrame, dx_nm: float, dy_nm: float, dz_nm: float) -> pd.DataFrame:
    """Add volume_um3 from syn_size if present."""
    if "syn_size" not in df.columns:
        return df
    df = df.copy()
    voxel_vol_um3 = (dx_nm * dy_nm * dz_nm) / 1e9
    df["volume_um3"] = df["syn_size"] * voxel_vol_um3
    return df

def normalized_from_raw(df_raw: pd.DataFrame,
                        pos_unit: str = "nm") -> pd.DataFrame:
    """
    Produce normalized schema including 0/1 'is_on_spine' and type {'E','I'}.
    Keeps original 0/1 coding (no booleans).
    """
    if "isOnSpine" not in df_raw.columns:
        raise KeyError("Expected 'isOnSpine' column in synapse table.")

    df_norm = normalize_position_columns(df_raw, x="x", y="y", z="z", unit=pos_unit)
    # Keep integer 0/1 flag
    df_norm["is_on_spine"] = df_raw["isOnSpine"].astype(int)
    df_norm["type"] = df_norm["is_on_spine"].map(lambda v: "E" if v == 0 else "I")

    keep = [c for c in ["x_um","y_um","z_um","syn_size","volume_um3","is_on_spine","type"] if c in df_norm.columns]
    return df_norm[keep]

def drop_duplicate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Exact dedupe on x_um,y_um,z_um (normalized layer)."""
    cols = [c for c in ["x_um","y_um","z_um"] if c in df.columns]
    if not cols:
        return df
    return df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)
