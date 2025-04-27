"""
This script updates EXIF metadata in image files for a given batch. It:
- Converts epoch-based filenames to EXIF-compliant timestamps.
- Adds additional EXIF tags from the config file.
- Uses multiprocessing for faster updates via `exiftool`.
"""

import datetime
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from shutil import which

import hydra
import pytz
from omegaconf import DictConfig

from src.utils.utils import find_lts_dir

log = logging.getLogger(__name__)

def epoch_to_datetime(epoch: int) -> str:
    """
    Convert epoch timestamp to standard string.
    Returns:
        str: Formatted date-time string.
    """
    try:
        dt = datetime.datetime.fromtimestamp(epoch)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        log.error(f"Error converting epoch to datetime: {e}")
        return ""

def epoch_to_exif_datetime_eastern(epoch: int, use_fractional=True) -> str:
    """
    Convert epoch timestamp to EXIF-compliant datetime string in US Eastern Time (auto DST).

    Args:
        epoch (int): Epoch timestamp
        use_fractional (bool): Include fractional seconds (2 digits)

    Returns:
        str: EXIF-compliant datetime string with Eastern timezone offset
    """
    eastern = pytz.timezone('US/Eastern')
    dt = datetime.fromtimestamp(epoch, tz=pytz.utc).astimezone(eastern)

    # Build datetime string
    base = dt.strftime("%Y:%m:%d %H:%M:%S")
    if use_fractional:
        fractional = f"{dt.microsecond // 10000:02d}"  # Keep 2 digits
        base += f".{fractional}"

    offset = dt.strftime("%z")  # e.g., -0400
    base += f"{offset[:3]}:{offset[3:]}"  # format as -04:00

    return base

def flatten_exif_dict(config: dict) -> dict:
    """Flatten a nested EXIF dictionary for exiftool usage."""
    items = {}
    for k, v in config.items():
        for k2, val in v.items():
            new_key = f"{k2}"
            if val is not None and val != "":
                items[f"{new_key}"] = val
    return items

def _update_exif_worker(args):
    """
    Worker function to update EXIF tags for one image.

    Args:
        args (tuple): (Path to image, flattened EXIF tags dict)
    """
    file_path, base_tags = args
    img_stem = file_path.stem
    try:
        epoch = int(img_stem.split("_")[1])
        datetime_original = epoch_to_exif_datetime_eastern(epoch)
    except Exception as e:
        log.error(f"Error parsing epoch for {file_path.name}: {e}")
        return

    tags = base_tags.copy()
    tags["SubSecDateTimeOriginal"] = f"'{datetime_original}'"

    cmd = ["exiftool", "-overwrite_original"]
    for key, value in tags.items():
        if isinstance(value, list):
            value = ",".join(map(str, value))
        cmd.append(f"-{key}={value}")
    cmd.append(str(file_path))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            log.info(f"[{file_path.name}] EXIF updated successfully.")
        else:
            log.error(f"[{file_path.name}] EXIF update failed with error: {result.stderr.strip()}")
        if result.stdout:
            log.debug(f"[{file_path.name}] STDOUT: {result.stdout.strip()}")
        if result.stderr:
            log.debug(f"[{file_path.name}] STDERR: {result.stderr.strip()}")
    except Exception as e:
        log.exception(f"[{file_path.name}] Exception during EXIF update: {e}")

def batch_update(cfg: DictConfig, image_dir: Path):
    """Update EXIF tags in all image files within a directory using multiprocessing."""
    tags = flatten_exif_dict(cfg.exif)
    if not tags:
        log.warning("No EXIF tags found in config. Skipping update.")
        return

    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        log.warning(f"No .jpg images found in {image_dir}")
        return
    
    log.info(f"Found {len(images)} images in {image_dir} for EXIF update.")
    
    args = [(img, tags) for img in images]

    with ProcessPoolExecutor(max_workers=16) as executor:
        list(executor.map(_update_exif_worker, args))


def ensure_exiftool_installed(setup_script_path: Path = Path("setup_exiftool.sh")):
    """Ensure exiftool is available. If not, run setup script to install it."""

    if which("exiftool"):
        log.info("ExifTool is available.")
        return

    log.warning("ExifTool not found in PATH. Attempting installation...")

    try:
        result = subprocess.run(
            ["bash", str(setup_script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        log.info(f"Setup script stdout:\n{result.stdout}")
        if result.stderr:
            log.warning(f"Setup script stderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        log.error(f"ExifTool installation failed: {e.stderr}")
        raise RuntimeError("ExifTool setup failed.") from e

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for updating exif information."""
    
    log.info("Running EXIF tag update pipeline...")
    batch_id = cfg.batch_id
    try:
        log.info("Checking for exiftool installation...")
        # Ensure exiftool is ready
        setup_script = Path(cfg.paths.workdir) / "scripts" / "setup_exiftool.sh"
        ensure_exiftool_installed(setup_script)
    except Exception as e:
        log.error(f"ExifTool setup failed: {e}")
        raise
    
    try:
        log.info(f"Finding image directory for batch ID: {batch_id}")
        # Find image directory
        lts_dir = find_lts_dir(batch_id, cfg.paths.lts_locations, local=False, developed=True, jpgs=True)
        image_directory = Path(lts_dir) / "semifield-developed-images" / batch_id / "images"
        if not image_directory.exists():
            log.error(f"Image directory not found: {image_directory}")
            return
    except Exception as e:
        log.error(f"Failed to find image directory: {e}")
        raise
    
    try:
        log.info(f"Updating EXIF tags in directory: {image_directory}")
        # Update EXIF tags
        batch_update(cfg, image_directory)
        log.info("Finished updating EXIF tags.")
    except Exception as e:
        log.error(f"Failed to update EXIF tags: {e}")
        raise
    
    log.info("EXIF update process completed successfully.")
    return

if __name__ == "__main__":
    main()
