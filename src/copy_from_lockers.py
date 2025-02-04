import os  
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import yaml
import sys
import logging
import hydra
from omegaconf import DictConfig
from datetime import datetime

log = logging.getLogger(__name__)


def copy_raw_file(src, dest, log: logging.Logger):
    """Copy a single RAW file from src to dest, if not already present with the same size."""
    if not os.path.exists(dest) or os.path.getsize(src) != os.path.getsize(dest):
        shutil.copy2(src, dest)
        log.info(f"Copied {src} to {dest}")
    else:
        log.info(f"Skipped {src}, already present at destination with matching size")
       
def copy_from_lockers_in_parallel(src_dir, dest_dir, log: logging.Logger, max_workers=12, raw_extension=".ARW"):
    """Copy all ARW files in parallel from NFS to local storage."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    if raw_extension.upper() == ".ARW":
        raw_extension = raw_extension.upper()

    elif raw_extension.upper() == ".RAW":
        raw_extension = raw_extension.upper()
    
    log.info(f"Copying raw image files with extension {raw_extension}")
    # Collect all ARW file paths
    raw_files = list(Path(src_dir).glob(f"*{raw_extension}"))
    
    log.info(f"Copying {len(raw_files)} raw image files from {src_dir} to {dest_dir}")
    
    # Use ThreadPoolExecutor for parallel copying
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_raw_file, arw, os.path.join(dest_dir, os.path.basename(arw)), log) for arw in raw_files]
        for future in as_completed(futures):
            try:
                future.result()  # Capture any exceptions
            except Exception as e:
                log.error(f"Error copying file: {e}")

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    
    batch_id = cfg.batch_id

    nfs_path = Path(cfg['paths']['primary_nfs'], "semifield-upload")
    local_uploads = Path(cfg['paths']['local_upload'])
    
    dst_dir = local_uploads / batch_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    src_dir = nfs_path / batch_id

    assert Path(src_dir).exists(), f"Source directory {src_dir} does not exist. Check the batch name."

    log.info(f"Copying from {src_dir} to {dst_dir}")

    copy_from_lockers_in_parallel(src_dir, dst_dir, log, raw_extension=cfg['copy_from_lockers']['raw_extension'])

if __name__ == "__main__":
    main()
