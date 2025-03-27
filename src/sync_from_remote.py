"""
This script ensures that local model and metadata files are up-to-date with their remote counterparts. 
It compares files based on their SHA-256 hash. If a local file is missing or differs from the remote 
version, it is automatically downloaded and updated.
"""

import logging
import hashlib
import shutil
from pathlib import Path
from omegaconf import DictConfig
import hydra

log = logging.getLogger(__name__)

def compute_file_hash(filepath: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash for a file."""
    hash_func = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def files_are_identical(local_file: Path, remote_file: Path) -> bool:
    """Compare hash of local and remote files to determine if they are identical."""
    if not local_file.exists() or not remote_file.exists():
        log.warning(f"Missing file: {local_file} or {remote_file}")
        return False

    local_hash = compute_file_hash(local_file)
    remote_hash = compute_file_hash(remote_file)
    
    return local_hash == remote_hash

def update_file(local_file: Path, remote_file: Path):
    """Copy remote file to local if they are different."""
    try:
        log.info(f"Updating {local_file} from {remote_file}")
        local_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_file, local_file)
        log.info(f"Update complete: {local_file}")
    except Exception as e:
        log.error(f"Failed to update {local_file}: {e}")

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Checks if local files are up to date compared to their remote versions,
    and updates them if necessary.
    """
    # TODO: add pp3 file, ground control points
    model_files = {
        "detection_model": {
            "local": Path(cfg.paths.local_detection_model),
            "remote": Path(cfg.paths.lts_detection_model),
        },
        "species_info": {
            "local": Path(cfg.paths.species_info),
            "remote": Path(cfg.paths.lts_species_info),
        }
    }

    for name, paths in model_files.items():
        local = paths["local"]
        remote = paths["remote"]

        if not remote.exists():
            log.warning(f"Remote file {remote} does not exist, skipping.")
            continue
        
        if not local.exists():
            log.warning(f"Local file {local} does not exist. Downloading...")
            update_file(local, remote)
            continue
        
        if files_are_identical(local, remote):
            log.info(f"{name}: Local file is up-to-date.")
        else:
            log.warning(f"Local {name} is outdated or different from the remote version. Updating...")
            update_file(local, remote)

if __name__ == "__main__":
    main()