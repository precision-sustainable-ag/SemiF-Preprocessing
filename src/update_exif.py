import datetime
from datetime import datetime
import pytz
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from shutil import which

import hydra
from omegaconf import DictConfig

from src.utils.utils import find_lts_dir

log = logging.getLogger(__name__)

def epoch_to_datetime(epoch: int) -> str:
    """Convert an epoch timestamp to 'YYYY:MM:DD HH:MM:SS' format."""
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

def flatten_exif_dict(config: dict, parent_key: str = '', sep: str = ':') -> dict:
    """Flatten a nested EXIF dictionary for exiftool usage."""
    items = {}
    for k, v in config.items():
        for k2, val in v.items():
            new_key = f"{k2}"
            if val is not None and val != "":
                items[f"{new_key}"] = val
    return items

def _update_exif_worker(args):
    """Worker function for multiprocessing."""
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
    log.info(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            log.warning(f"Command output: {result.stdout}")
        if result.stderr:
            log.error(f"Command error: {result.stderr}")
        if result.returncode == 0:
            log.info(f"Updated: {file_path.name}")
        else:
            log.error(f"Failed: {file_path.name}\n{result.stderr}")
    except Exception as e:
        log.error(f"Error updating {file_path.name}: {e}")

def batch_update(cfg: DictConfig, image_dir: Path):
    """Update EXIF tags in all image files within a directory using multiprocessing."""
    tags = flatten_exif_dict(cfg.exif)
    if not tags:
        log.warning("No EXIF tags loaded. Aborting.")
        return

    images = sorted(image_dir.glob("*.jpg"), reverse=False)
    args = [(img, tags) for img in images]
    multiprocess = True
    if multiprocess:
        with ProcessPoolExecutor(max_workers=16) as executor:
            list(executor.map(_update_exif_worker, args))
    else:
        for arg in args:
        # arg = (Path("/mnt/research-projects/s/screberg/longterm_images2/semifield-developed-images/NC_2025-03-17/images/NC_1742223170.jpg"), tags)
            _update_exif_worker(arg)

def ensure_exiftool_installed(setup_script_path: Path = Path("setup_exiftool.sh")):
    """Ensure exiftool is available. If not, run setup script to install it."""

    exiftool_path = which("exiftool")
    if exiftool_path:
        log.info(f"ExifTool found at: {exiftool_path}")
        return

    log.warning("ExifTool not found in PATH. Running setup script...")

    try:
        result = subprocess.run(["bash", str(setup_script_path)], capture_output=True, text=True, check=True)
        log.info(result.stdout)
        log.warning(result.stderr)
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to run setup script: {e.stderr}")
        raise RuntimeError("ExifTool setup failed.") from e

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for updating exif information."""
    log.info("Starting update_exif_tags.py")
    batch_id = cfg.batch_id
    # Check for exiftool
    setup_script = Path(cfg.paths.workdir) / "scripts" / "setup_exiftool.sh"
    ensure_exiftool_installed(setup_script)
    # Find image directory
    lts_dir = find_lts_dir(batch_id, cfg.paths.lts_locations, local=False, developed=True, jpgs=True)
    image_directory = Path(lts_dir) / "semifield-developed-images" / batch_id / "images"
    # Update EXIF tags
    batch_update(cfg, image_directory)
    log.info("Finished updating EXIF tags.")

if __name__ == "__main__":
    main()
