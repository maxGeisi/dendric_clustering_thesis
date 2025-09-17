# src/processing/branch_processing.py
"""
Branch processing functions for dendric clustering analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import navis
from collections import Counter
from typing import Dict, List, Any, Tuple

def split_branches(neuron_skel) -> Tuple[List[Any], List[int]]:
    """
    Split neuron at all branchpoints.
    
    Args:
        neuron_skel: Navis neuron skeleton object
        
    Returns:
        Tuple of (neuron_splits, junctions_list)
    """
    # Split neuron at all branchpoints
    n_parents = neuron_skel.nodes.parent_id
    # Get each parent, that is not unique
    counts = Counter(n_parents)
    junctions_list = [item for item, count in counts.items() if count > 1]
    
    # Take the nodes that have this parent as their parent_id (thus child)
    junctions_list = neuron_skel.nodes[neuron_skel.nodes.parent_id.isin(junctions_list)].node_id.tolist()
    n_branches = navis.cut_skeleton(neuron_skel, junctions_list)
    
    return n_branches, junctions_list

def build_branch_dataframe(
    neuron_splits: List[Any],
    syn_exec_df: pd.DataFrame,
    syn_inh_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build branch DataFrame with synapse statistics.
    
    Args:
        neuron_splits: List of branch skeleton objects
        syn_exec_df: DataFrame with excitatory synapse data
        syn_inh_df: DataFrame with inhibitory synapse data
        
    Returns:
        DataFrame with branch statistics
    """
    branch_stats = []
    
    # Debug: Print available columns
    print(f"   • syn_exec_df columns: {list(syn_exec_df.columns)}")
    print(f"   • syn_inh_df columns: {list(syn_inh_df.columns)}")
    
    # Determine the correct column names
    exec_id_col = 'Exec_syn_id' if 'Exec_syn_id' in syn_exec_df.columns else 'id'
    inh_id_col = 'Inh_syn_id' if 'Inh_syn_id' in syn_inh_df.columns else 'id'
    
    print(f"   • Using exec_id_col: {exec_id_col}")
    print(f"   • Using inh_id_col: {inh_id_col}")
    
    for sk in neuron_splits:
        root_node = sk.nodes.loc[sk.nodes['type'] == 'root', 'node_id'].iat[0]
        node_ids = sk.nodes['node_id'].tolist()
        
        e_syn_ids = syn_exec_df.loc[
            syn_exec_df['closest_node_id'].isin(node_ids), exec_id_col
        ].tolist()
        
        i_syn_ids = syn_inh_df.loc[
            syn_inh_df['closest_node_id'].isin(node_ids), inh_id_col
        ].tolist()
        
        branch_stats.append({
            'junction_id': root_node,
            'branch_length': sk.cable_length,
            'n_e': len(e_syn_ids),
            'n_i': len(i_syn_ids),
            'e_syn_ids': e_syn_ids,
            'i_syn_ids': i_syn_ids
        })
    
    # Build DataFrame
    branch_df = pd.DataFrame(branch_stats)
    
    # Add 1-based sequential branch_idx
    branch_df.insert(0, 'branch_idx', range(0, len(branch_df)))
    
    return branch_df

def filter_branches_by_length(
    branch_df: pd.DataFrame,
    percentile_cutoff: float = 0.05
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filter branches by length using percentile cutoff.
    
    Args:
        branch_df: DataFrame with branch information
        percentile_cutoff: Percentile cutoff (default 0.05 for 5th percentile)
        
    Returns:
        Tuple of (filtered_branch_df, filter_stats)
    """
    # Compute the percentile cutoff
    cutoff = branch_df['branch_length'].quantile(percentile_cutoff)
    
    # Identify branches to remove
    too_small = branch_df[branch_df['branch_length'] < cutoff]
    
    # Count synapses lost
    n_e_lost = too_small['n_e'].sum()
    n_i_lost = too_small['n_i'].sum()
    
    # Filter statistics
    filter_stats = {
        'cutoff': cutoff,
        'branches_removed': len(too_small),
        'total_branches': len(branch_df),
        'e_synapses_lost': n_e_lost,
        'i_synapses_lost': n_i_lost
    }
    
    # Drop small branches
    filtered_branch_df = branch_df[branch_df['branch_length'] >= cutoff].reset_index(drop=True)
    
    return filtered_branch_df, filter_stats

def add_volume_metrics(
    branch_df: pd.DataFrame,
    syn_exec_df: pd.DataFrame,
    syn_inh_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add volume metrics to branch DataFrame.
    
    Args:
        branch_df: DataFrame with branch information
        syn_exec_df: DataFrame with excitatory synapse data
        syn_inh_df: DataFrame with inhibitory synapse data
        
    Returns:
        DataFrame with added volume metrics
    """
    branch_df = branch_df.copy()
    
    # Determine the correct column names
    exec_id_col = 'Exec_syn_id' if 'Exec_syn_id' in syn_exec_df.columns else 'id'
    inh_id_col = 'Inh_syn_id' if 'Inh_syn_id' in syn_inh_df.columns else 'id'
    
    # Sum volumes per branch
    branch_df['exec_vol_sum'] = branch_df['e_syn_ids'].apply(
        lambda ids: syn_exec_df.loc[
            syn_exec_df[exec_id_col].isin(ids), 'SynapseVolume'
        ].sum() if ids else 0
    )
    
    branch_df['inh_vol_sum'] = branch_df['i_syn_ids'].apply(
        lambda ids: syn_inh_df.loc[
            syn_inh_df[inh_id_col].isin(ids), 'SynapseVolume'
        ].sum() if ids else 0
    )
    
    branch_df['total_syn'] = branch_df['n_e'] + branch_df['n_i']
    
    return branch_df

def get_branches_with_both_synapse_types(branch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter branches that have both excitatory and inhibitory synapses.
    
    Args:
        branch_df: DataFrame with branch information
        
    Returns:
        DataFrame with branches that have both synapse types
    """
    mask = (branch_df['e_syn_ids'].map(len) > 0) & (branch_df['i_syn_ids'].map(len) > 0)
    return branch_df[mask].copy()

def get_branch_synapses(
    branch_idx: int,
    neuron_splits: List[Any],
    syn_exec_df: pd.DataFrame,
    syn_inh_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get synapses for a specific branch.
    
    Args:
        branch_idx: Index of the branch
        neuron_splits: List of branch skeleton objects
        syn_exec_df: DataFrame with excitatory synapse data
        syn_inh_df: DataFrame with inhibitory synapse data
        
    Returns:
        Tuple of (excitatory_synapses_df, inhibitory_synapses_df)
    """
    rand_branch = neuron_splits[branch_idx]
    node_ids = rand_branch.nodes["node_id"].tolist()
    
    df_e_branch = syn_exec_df[syn_exec_df["closest_node_id"].isin(node_ids)]
    df_i_branch = syn_inh_df[syn_inh_df["closest_node_id"].isin(node_ids)]
    
    return df_e_branch, df_i_branch

def print_branch_filtering_summary(filter_stats: Dict[str, Any]) -> None:
    """
    Print summary of branch filtering results.
    
    Args:
        filter_stats: Dictionary with filtering statistics
    """
    print(f"5th-percentile cutoff: {filter_stats['cutoff']:.2f} μm")
    print(f"Removing {filter_stats['branches_removed']} branches (out of {filter_stats['total_branches']})")
    print(f"→ E-synapses lost: {filter_stats['e_synapses_lost']}")
    print(f"→ I-synapses lost: {filter_stats['i_synapses_lost']}")
    print(f"Remaining branches: {filter_stats['total_branches'] - filter_stats['branches_removed']}")

