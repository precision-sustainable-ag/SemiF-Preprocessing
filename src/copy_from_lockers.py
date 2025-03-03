import os  
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
from omegaconf import DictConfig
from src.utils.utils import find_lts_dir

log = logging.getLogger(__name__)

def copy_raw_file(src, dest):
    """Copy a single RAW file from src to dest, if not already present with the same size."""
    if not os.path.exists(dest) or os.path.getsize(src) != os.path.getsize(dest):
        shutil.copy2(src, dest)
        log.info(f"Copied {src} to {dest}")
    else:
        log.info(f"Skipped {src}, already present at destination with matching size")
       
def copy_from_lockers_in_parallel(src_dir: Path, dest_dir: Path, max_workers:int=12, raw_extension: str=".ARW") -> None:
    """
    Copy all ARW files in parallel from NFS to local storage.

    Args:
        src_dir (Path): Path to the batch folder in LTS location.
        dest_dir (Path): local folder to copy data to.
        max_workers (int, optional): Number of parallel workers to use. Defaults to 12.
        raw_extension (str, optional): File extension to use. Defaults to ".ARW".
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    if raw_extension.upper() == ".ARW":
        raw_extension = raw_extension.lower()

    elif raw_extension.upper() == ".RAW":
        raw_extension = raw_extension.upper()
    
    log.info(f"Copying raw image files with extension {raw_extension}")
    # Collect all ARW file paths
    raw_files = sorted(list(src_dir.glob(f"*{raw_extension}")))

    # Optionally filter files by specific names.
    # raw_files = [f for f in raw_files if f.stem in ["NC_1740166530", "NC_1740167524", "NC_1740162656"]] 

    log.info(f"Copying {len(raw_files)} raw image files from {src_dir} to {dest_dir}")
    
    # Use ThreadPoolExecutor for parallel copying
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_raw_file, arw, os.path.join(dest_dir, os.path.basename(arw))) for arw in raw_files]
        for future in as_completed(futures):
            try:
                future.result()  # Capture any exceptions
            except Exception as e:
                log.error(f"Error copying file: {e}")
    return

def main(cfg: DictConfig) -> None:
    """
    Entrypoint for copying RAW files from NFS to local storage:
    <lts_location>/semifield-upload/<batch_id> to
    ./data/<lts_location>/semifield-upload/<batch_id>
    """
    log.info(f"Copying RAW files from NFS to local storage")
    batch_id = cfg.batch_id

    lts_locations = cfg.paths.lts_locations
    nfs_path = find_lts_dir(batch_id, lts_locations)
    local_uploads = Path(cfg.paths.data_dir, nfs_path.name, "semifield-upload")
    
    dst_dir = local_uploads / batch_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    src_dir = nfs_path / "semifield-upload" / batch_id

    assert src_dir.exists(), f"Source directory {src_dir} does not exist. Check the batch name."

    log.info(f"Copying from {src_dir} to {dst_dir}")

    copy_from_lockers_in_parallel(src_dir, dst_dir, raw_extension=".RAW")