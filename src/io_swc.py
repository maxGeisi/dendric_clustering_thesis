from pathlib import Path
import navis

def _project_root() -> Path:
    here = Path.cwd()
    for p in (here, *here.parents):
        if (p / "src").exists() and (p / "config").exists():
            return p
    return Path.cwd()

def build_swc_path(swc_dir: str | Path, prefix: str, neuron_id: str) -> Path:
    swc_dir = Path(swc_dir)
    if not swc_dir.is_absolute():
        swc_dir = _project_root() / swc_dir
    p = swc_dir / f"{prefix}{neuron_id}.swc"
    if not p.exists():
        raise FileNotFoundError(f"SWC not found: {p}")
    return p


def load_and_heal_swc(swc_path: str | Path):
    import navis
    """
    Read SWC and return a healed skeleton neuron.
    Works if navis returns NeuronList (takes first) or a single neuron.
    """
    n = navis.read_swc(str(swc_path))
    if hasattr(n, "__iter__") and not hasattr(n, "nodes"):
        n = n[0]
    n = navis.heal_skeleton(n)
    return n
