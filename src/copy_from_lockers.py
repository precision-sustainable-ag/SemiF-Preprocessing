import os
import shutil
import random
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union

import hydra
from omegaconf import DictConfig

from utils.utils import find_lts_dir

log = logging.getLogger(__name__)


def copy_raw_file(src: Union[str, os.PathLike], dest: Union[str, os.PathLike]) -> None:
    """
    Copy a single RAW file from src to dest, if not already present with the same size.
    """
    # Check if the destination file doesn't exist or sizes differ
    if not os.path.exists(dest) or os.path.getsize(src) != os.path.getsize(dest):
        shutil.copy2(src, dest)
        log.info(f"Copied {src} to {dest}")
    else:
        log.info(f"Skipped {src}, already present at destination with matching size")


def copy_from_lockers_in_parallel(
    src_dir: Union[str, os.PathLike],
    dest_dir: Path,
    test_cfg: DictConfig,
    max_workers: int = 12
) -> None:
    """
    Copy all RAW files in parallel from the source directory to the destination directory.

    Uses ThreadPoolExecutor for parallel copying. If testing is enabled, only a sample of files 
    are copied to speed up the process.

    Args:
        src_dir (Union[str, os.PathLike]): Source directory containing RAW files.
        dest_dir (Path): Destination directory where files will be copied.
        test_cfg (DictConfig): Testing configuration with attributes 'enabled' and 'sample_size'.
        max_workers (int, optional): Maximum number of parallel workers. Defaults to 12.
    """
    # Create the destination directory if it doesn't exist
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)

    # Collect all RAW file paths from the source directory
    raw_files = sorted(list(Path(src_dir).glob("*.RAW")))[:10]

    # Optionally sample a subset of files for testing purposes
    if test_cfg.enabled:
        log.info(f"Testing enabled. Sampling {test_cfg.sample_size} files from {len(raw_files)}")
        raw_files = random.sample(raw_files, min(test_cfg.sample_size, len(raw_files)))

    log.info(f"Found {len(raw_files)} RAW files in {src_dir}")

    # Filter out files that already exist at the destination with matching sizes
    raw_files = [
        file for file in raw_files
        if not (dest_dir / file.name).exists() or os.path.getsize(file) != os.path.getsize(dest_dir / file.name)
    ]

    if not raw_files:
        log.info("No new RAW files to copy")
        return

    log.info(f"Found {len(raw_files)} new RAW files to copy")

    # Use ThreadPoolExecutor to copy files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit copying tasks for each file
        futures = [
            executor.submit(copy_raw_file, arw, os.path.join(dest_dir, os.path.basename(arw)))
            for arw in raw_files
        ]
        # Process each future as it completes, logging any exceptions
        for future in as_completed(futures):
            try:
                future.result()  # This will re-raise any exceptions caught during file copy
            except Exception as e:
                log.error(f"Error copying file: {e}")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to execute the RAW file copying workflow.

    Determines the correct source directory based on the batch ID, creates necessary 
    local directories, and initiates the parallel copying process.
    """
    log.info(f"Starting RAW file copying workflow for batch {cfg.batch_id}")
    test_cfg = cfg.test
    batch_id = cfg.batch_id

    # Locate the LTS directory for the given batch_id; exit if not found or incomplete
    lts_dir = find_lts_dir(batch_id, cfg.paths.lts_locations)
    if not lts_dir:
        log.error(f"Either batch {batch_id} not found or RAW files are incomplete. Exiting.")
        return

    # Define the source directory for the RAW files
    batch_lts_src_dir = Path(lts_dir, "semifield-upload", batch_id)

    # Define and create the destination directory for local uploads
    local_uploads_dst = Path(cfg.paths.data_dir) / lts_dir.name / "semifield-upload" / batch_id
    local_uploads_dst.mkdir(parents=True, exist_ok=True)

    log.info(f"Copying from {batch_lts_src_dir} to {local_uploads_dst}")

    # Start the parallel copying of RAW files
    copy_from_lockers_in_parallel(batch_lts_src_dir, local_uploads_dst, test_cfg)

    log.info("RAW file copying workflow complete.")


if __name__ == "__main__":
    main()
