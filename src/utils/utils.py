from pathlib import Path
from PIL import Image
import logging
import piexif
import math
import numpy as np
import yaml
import shutil
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import re
from typing import Dict, Any, List
import subprocess
import json
import os
import time

log = logging.getLogger(__name__)

def is_reconstructed(cfg: DictConfig) -> bool:
    """
    Check if the batch is reconstructed based on the presence of 'autosfm' in the artifact YAML.
    Raises an error if reconstruction status cannot be determined.
    """
    with open(cfg.paths.artifact_path, 'r') as f:
        artifact_data = yaml.safe_load(f)
    
    task_statuses = artifact_data.get("task_status", {})

    autosfm_status = task_statuses.get("autosfm", {})
    remap_labels_status = task_statuses.get("remap_labels", {})
    no_remap_label_status = task_statuses.get("no_remap_label", {})

    if no_remap_label_status == "success":
        log.info("No remap label task detected. Using simple labels.")
        return False

    if remap_labels_status == "success" and no_remap_label_status != "success" and autosfm_status == "success":
        log.info("Remap label task detected. Using reconstructed labels.")
        return True

    raise RuntimeError("Cannot determine reconstruction status from artifact YAML.")


def extract_season_info(cfg: DictConfig):
    # Parse batch id
    try:
        state_id, date_str = cfg.batch_id.split("_")
        batch_date = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception as e:
        raise ValueError("Batch ID format must be SITE_YYYY-MM-DD") from e

    # Scan through site's seasons
    date_ranges = cfg.date_ranges
    for state_name, info in date_ranges.items():
        if state_id == state_name:
            for k, v in info.items():
                start = datetime.strptime(v["start"], "%Y-%m-%d")
                end = datetime.strptime(v["end"], "%Y-%m-%d")
                if start <= batch_date <= end:
                    cfg.season = v["pipeline_season"] if cfg.season is None else cfg.season
                    cfg.bbot_version = v["bbot_version"] if cfg.bbot_version is None else cfg.bbot_version
                    cfg.crs = v["crs"] if cfg.crs is None else cfg.crs
                    cfg.gh_reviewer = v.get("gh_reviewer", "mkutu")

    return cfg

def replace_all_nan_with_null(obj: Any) -> Any:
    """
    Recursively replace all NaN values with None (JSON null).
    """
    if isinstance(obj, dict):
        return {k: replace_all_nan_with_null(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_all_nan_with_null(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj

def save_json_file(file_path: Path, data: Dict[str, Any]) -> None:
    """
    Save a dictionary to a JSON file.
    """
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
        
def safe_save_json(data: Any, path: Path) -> None:
    cleaned = replace_all_nan_with_null(data)
    save_json_file(path, cleaned)

def find_batches(lts_dirs: List[Path]) -> List[Path]:
    """
    Find all batches in the given directories.
    """
    batches = []
    for lts_dir in lts_dirs:
        batches.extend(sorted(lts_dir.glob("*")))
    batches = [batch for batch in batches if matches_batch_format(batch.name)]
    return batches

def get_json_files(batch: Path) -> List[Path]:
    """
    Get all JSON files in the given batch directory.
    """
    json_dir = batch / "metadata"
    return sorted(json_dir.glob("*.json"))

def backup_metadata(batch: Path, local_test_dir: Path) -> None:
    """
    Backup metadata for the given batch to the local batch directory.
    """
    local_batch_dir = local_test_dir / batch.parent.parent.name / "semifield-developed-images" / batch.name
    # Create the local batch directory if it doesn't exist
    local_batch_dir.mkdir(parents=True, exist_ok=True)
    # Backup the original JSON file data locally
    shutil.copytree(batch / "metadata", local_batch_dir, dirs_exist_ok=True)

def matches_batch_format(batch: str) -> bool:
    """
    Check if the batch name matches the expected format. Which is NC, MD, or TX followed by a date in the format of 2024-01-01. A _ separates state and date.
    """
    pattern = r"^(NC|MD|TX)_(\d{4}-\d{2}-\d{2})$"
    return bool(re.match(pattern, batch))

def strict_parse_constant(val: str):
    raise ValueError(f"Invalid constant found in JSON: {val}")

def read_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Read a JSON file and return its content.
    """
    with open(file_path, "r") as file:
        data = json.load(file)#, parse_constant=strict_parse_constant)

    return data



def set_cpu_affinity() -> None:
    try:
        os.sched_setaffinity(0, set(range(2, 32)))
        log.info("Set CPU affinity to cores 2-31")
    except AttributeError:
        log.warning("CPU affinity setting not supported on this platform.")
    except Exception as e:
        log.warning(f"Failed to set CPU affinity: {e}")

def find_raw_dir(local_data_dir: Path, batch_id: str,
                 lts_dir: Path) -> Path | None:
    """Find the raw directory containing .RAW files, preferring the one with more files."""

    def count_raw_files(directory: Path) -> int:
        """Return the count of .RAW files if the directory exists, otherwise 0."""
        return len(list(directory.glob("*.RAW"))) if directory.exists() else 0

    local_raw_dir = Path(local_data_dir, lts_dir.name, "semifield-upload",
                         batch_id)
    remote_raw_dir = Path(lts_dir, "semifield-upload", batch_id)

    if not remote_raw_dir.exists():
        log.error(f"Remote RAW directory not found: {remote_raw_dir}. Exiting.")
        raise FileNotFoundError(
            f"Remote RAW directory not found: {remote_raw_dir}")

    # if local does not exist, return remote
    if not local_raw_dir.exists():
        return remote_raw_dir

    local_count = count_raw_files(local_raw_dir)
    remote_count = count_raw_files(remote_raw_dir)

    if local_count > 0 or remote_count > 0:
        if local_count >= remote_count:
            log.info(
                f"Using local RAW directory: {local_raw_dir} ({local_count} files)")
            return local_raw_dir
        else:
            log.info(
                f"Using remote RAW directory: {remote_raw_dir} ({remote_count} files)")
            return remote_raw_dir

    log.warning(f"No RAW directory found for batch {batch_id}")
    return None


# Find the batch NFS location from a list of possible parent directories
def find_lts_dir(batch_id: str, nfs_locations: list[str], local: bool = False,
                 developed: bool = False, dngs: bool = False, jpgs: bool = False) -> Path | None:
    """
    Searches for the specified batch directory within the given NFS locations and checks for the presence and completeness of RAW files.
    Args:
        batch_id (str): The identifier of the batch to search for.
        nfs_locations (list): A list of NFS locations (directories) to search within.
        local (bool): true - searches for batch data in local directory.
        developed (bool): true - searches for pngs in semifield-developed-images, false - searches for raws in semifield-upload.
    Returns:
        Path: The NFS location where the batch was found with complete RAW
        files, or None if the batch is not found or the files are incomplete.
    Logs:
        - Info: Logs the NFS location and the number of RAW files found if the batch is found and the files are complete.
        - Error: Logs an error message if the batch is not found, if no RAW files are found, or if the RAW files are not completely uploaded.
    """
    dir_found, files_found, upload_complete = False, False, False
    batch_location = None
    for nfs_location in nfs_locations:
        nfs_location = Path(nfs_location)
        if local:
            if developed:
                batch_location = Path(
                    "data") / nfs_location.name / "semifield-developed-images" / batch_id
            else:
                batch_location = Path(
                    "data") / nfs_location.name / "semifield-upload" / batch_id
        else:
            if developed:
                batch_location = nfs_location / "semifield-developed-images" / batch_id
            else:
                batch_location = nfs_location / "semifield-upload" / batch_id
        # Check if the batch directory exists
        if batch_location.exists():
            dir_found = True
            if developed:
                if dngs:
                    dng_location = batch_location / "dngs"
                    if dng_location.exists():
                        return nfs_location
                elif jpgs:
                    files = list(Path(batch_location, "images").glob("*.jpg")) + list(
                        Path(batch_location, "images").glob("*.JPG"))
                else:
                    files = list(Path(batch_location, "pngs").glob("*.png")) + list(
                        Path(batch_location, "pngs").glob("*.PNG"))
            else:
                files = list(batch_location.glob("*.RAW")) + list(
                    batch_location.glob("*.raw"))
            # Check if any RAW files are present
            if files:
                # todo: md5 checksum for data verification?
                log.info(
                    f"Batch {batch_id} found in {batch_location} with {len(files)} {'RAW' if not developed else ('JPG' if jpgs else 'PNG')} files")
                return nfs_location
    if not dir_found:
        log.error(
            f"Batch {batch_id} not found in NFS locations: {nfs_locations}")
    elif not files_found:
        log.error(
            f"Batch {batch_id} found in {batch_location} but no RAW files found")
    elif not upload_complete:
        log.error(
            f"Batch {batch_id} found in {batch_location} but RAW files are not completely uploaded")
    return None


def log_image_stats(image: np.ndarray, label: str = "Image"):
    log.debug(
        f"{label} - dtype: {image.dtype}, range: [{np.min(image)}, {np.max(image)}], shape: {image.shape}")


def estimate_focal_length_35mm(focal_length: int, sensor_height: float,
                               sensor_width: float) -> float:
    """
    Calculate the 35mm focal length based on sensor dimensions.
    """
    # Diagonal size of a 35mm full-frame sensor
    diag_35mm = math.sqrt(
        36 ** 2 + 24 ** 2)  # Full-frame diagonal in mm (43.27 mm)

    # Diagonal size of the given sensor
    diag_sensor = math.sqrt(sensor_width ** 2 + sensor_height ** 2)

    # Estimate Focal Length in 35mm Film format
    focal_length_35mm = focal_length * (diag_35mm / diag_sensor)
    return focal_length_35mm


def add_exif_data(image_path: Path, updated_exif: dict) -> None:
    """
    Function to add exif information to jpeg image
    """
    if ".jpg" not in image_path.name.lower():
        log.error(f"{image_path.name} is not a valid jpg")
        return

    if 'FocalLengthIn35mmFilm' not in updated_exif.keys():
        updated_exif['FocalLengthIn35mmFilm'] = estimate_focal_length_35mm(
            updated_exif['FocalLength'], updated_exif['SensorHeight'],
            updated_exif['SensorWidth'])
    try:
        exif_dict = piexif.load(str(image_path))
    except Exception:
        log.error(f"Could not load {image_path.name}")
        return

    # update focal length information
    exif_dict["Exif"][piexif.ExifIFD.FocalLength] = (
    int(updated_exif['FocalLength'] * 100), 100)  # Rational number
    exif_dict["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = int(
        updated_exif['FocalLengthIn35mmFilm'])

    # Update Image Dimensions
    exif_dict["0th"][piexif.ImageIFD.ImageLength] = int(
        updated_exif['ImageHeight'])
    exif_dict["0th"][piexif.ImageIFD.ImageWidth] = int(
        updated_exif['ImageWidth'])
    # use PIL to save the image with updated exif information
    image = Image.open(image_path)
    image.save(image_path, "jpeg", exif=piexif.dump(exif_dict), quality='keep', subsampling='keep')
    return

def read_yaml(yaml_path):
    try:
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        raise FileNotFoundError(f"File does not exist : {yaml_path}")

def save_log_to_lts(cfg):
    try:
        batch_id = cfg.batch_id
        log_src_path = Path(HydraConfig.get().runtime.output_dir) / f"{batch_id}.log"
        log_dst_dir = Path(find_lts_dir(batch_id, cfg.paths.lts_locations)) / "semifield-developed-images" / batch_id / "inspection"
        log_dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(log_src_path, log_dst_dir)
    except Exception as e:
        log.error(f"Failed to save log file: {e}")

def save_yaml_to_lts(cfg, lts_dev_dir):
    try:
        yaml_path = Path(cfg.paths.artifact_path)
        batch_id = cfg.batch_id
        yaml_dst_dir = Path(lts_dev_dir) / batch_id / "inspection"
        yaml_dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(yaml_path, yaml_dst_dir)
    except Exception as e:
        log.error(f"Failed to save YAML file: {e}")

def create_issue(cfg, issue_type, tsk: str = None, error_msg: str = None):
    batch_id = cfg.batch_id
    user_id = cfg.gh_reviewer
    lts_dev_dir = cfg.paths.lts_developed_directory
    if not lts_dev_dir:
        lts_dev_dir = Path(find_lts_dir(batch_id, cfg.paths.lts_locations, developed=True, jpgs=True)) / "semifield-developed-images"
    lts_dev_dir_name = Path(lts_dev_dir).parent.name

    save_yaml_to_lts(cfg, lts_dev_dir)

    if issue_type == "report":
        trigger_payload = {
                "event_type": "report-generated",
                "client_payload": {
                    "batch_id": batch_id,
                    "assignee": user_id,  # from cfg.report.reviewers
                    "lts_developed": lts_dev_dir_name
                }
            }
        
    elif issue_type == "failure":
        trigger_payload = {
                "event_type": "failure-reported",
                "client_payload": {
                    "batch_id": batch_id,
                    "assignee": user_id,
                    "task_name": tsk,
                    "error_msg": error_msg,
                    "lts_developed": lts_dev_dir_name
                }
            }
    subprocess.run([
                "curl", "-X", "POST", "https://api.github.com/repos/precision-sustainable-ag/SemiF-Preprocessing/dispatches",
                "-H", f"Authorization: token {os.environ['GITHUB_PAT']}",
                "-H", "Accept: application/vnd.github.v3+json",
                "-d", json.dumps(trigger_payload)
            ], check=True)


def retry_nfs_access(path: Path, 
                     mode: str = "read", 
                     retries: int = 5, 
                     delay: float = 2.0,
                     backoff: float = 1.5) -> bool:
    """
    Retry access to a Path (NFS) multiple times if PermissionError or OSError occurs.

    Args:
        path (Path): Path object pointing to a file or directory.
        mode (str): "read" (check existence/readability) or "write" (try writing a temp file).
        retries (int): Max number of retries.
        delay (float): Initial delay between retries in seconds.
        backoff (float): Backoff multiplier to increase delay.

    Returns:
        bool: True if access eventually succeeds, False otherwise.
    """
    assert mode in ["read", "write"], "mode must be 'read' or 'write'"

    for attempt in range(retries):
        try:
            if mode == "read":
                if not path.exists():
                    raise FileNotFoundError(f"{path} does not exist")
                if path.is_dir():
                    _ = list(path.iterdir())  # trigger PermissionError if any
                else:
                    _ = path.read_bytes()[:1]  # just try to read a byte

            elif mode == "write":
                test_file = path / ".nfs_test"
                test_file.write_text("test")
                test_file.unlink()

            log.info(f"NFS access succeeded on attempt {attempt+1}: {path}")
            return True

        except (PermissionError, OSError) as e:
            log.warning(f"Attempt {attempt+1} failed to access {path}: {e}")
            time.sleep(delay)
            delay *= backoff

    log.error(f"NFS access failed after {retries} attempts: {path}")
    return False
