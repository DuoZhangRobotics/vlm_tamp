"""Machine-local executable selection, independent of numerical tool settings."""
import os
from pathlib import Path


def simulator_python(explicit=None):
    """Prefer an explicit interpreter, then the isolated local simulator install."""
    selected = explicit or os.environ.get("VLM_TAMP_SIM_PYTHON")
    if selected:
        return str(Path(selected).expanduser())
    local = Path(__file__).resolve().parents[2] / ".runtime/envs/vlm-tamp/bin/python"
    if local.is_file():
        return str(local)
    # Retain the original workstation default for existing installations.
    return "/home/duo/miniforge3/envs/vlm-tamp/bin/python"
