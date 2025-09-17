from pathlib import Path
import random, numpy as np

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_data_directories(base_data_dir: str | Path, dataset_id: str) -> dict:
    """
    Ensure all necessary data directories exist for a given dataset.
    
    Args:
        base_data_dir: Base data directory (e.g., "data")
        dataset_id: Dataset identifier (e.g., "microns", "iascone")
        
    Returns:
        Dictionary with created directory paths
    """
    base_data_dir = Path(base_data_dir)
    dataset_dir = base_data_dir / dataset_id
    
    # Define all required directories
    directories = {
        'dataset_root': dataset_dir,
        'extracted_swc': dataset_dir / "extracted_swc",
        'swc_skel': dataset_dir / "extracted_swc" / "SWC_Skel",
        'syn': dataset_dir / "extracted_swc" / "syn",
        'interim_results': dataset_dir / "intirim_results"
    }
    
    # Create all directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
