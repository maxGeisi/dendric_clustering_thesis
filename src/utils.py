from pathlib import Path
import random, numpy as np

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
