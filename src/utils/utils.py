# Standard library
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import time
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np
import pandas as pd
import piexif
import yaml
from PIL import Image
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def sanitize_time_for_path(time_str: str) -> str:
    """Replace characters illegal in filesystem paths."""
    return str(time_str).replace(":", "-").replace(" ", "_")

def check_z_axis(z_axis_str: str) -> str:
    """
    Check and normalize the Z-axis string.
    """
    if not z_axis_str:
        return ""
    z_axis_str = z_axis_str.strip().lower()
    if z_axis_str.endswith("cm"):
        z_axis_str = z_axis_str[:-len("cm")].strip()
    return z_axis_str

def check_cam_angle(cam_angle_str: str) -> str:
    """
    Check and normalize the camera angle string.
    """
    # check if cam angle is none
    if not cam_angle_str:
        return ""
    cam_angle_str = cam_angle_str.strip().lower()
    if cam_angle_str.endswith("degrees"):
        cam_angle_str = cam_angle_str[:-len("degrees")].strip()
    if cam_angle_str.endswith("degree"):
        cam_angle_str = cam_angle_str[:-len("degree")].strip()

    return cam_angle_str

def filter_files_by_timestamp(files: list, start_epoch: int, end_epoch: int, image_ids: bool = False) -> list[tuple[Path, bool]]:
    """
    Filter files based on a start and end time stamp
    """
    filtered_files = []
    for path in files:
        # Extract the timestamp from the file name
        file_timestamp = Path(path).stem.split("_")[-1]
        # Check if the timestamp is in the file name
        if len(file_timestamp) == 10 and file_timestamp.isdigit():
            # Compare with the input timestamp
            if int(file_timestamp) >= start_epoch and int(file_timestamp) <= end_epoch:
                filtered_files.append(path)
    return filtered_files

def normalize_time_str(time_str: str) -> str:
    """
    Detects if the time string is in 12-hour (AM/PM) or 24-hour format
    and normalizes it into 24-hour 'HH:MM:SS'.
    
    Accepts both '1:25:18 PM' and '1:25:18PM'.
    """
    time_str = time_str.strip().upper()

    # Fix cases like '1:25:18PM' -> '1:25:18 PM'
    time_str = re.sub(r'(?<=\d)(AM|PM)$', r' \1', time_str)

    try:
        # Try parsing as 12-hour format (AM/PM)
        dt = datetime.strptime(time_str, "%I:%M:%S %p")
    except ValueError:
        try:
            # Try parsing as 24-hour format
            dt = datetime.strptime(time_str, "%H:%M:%S")
        except ValueError as e:
            raise ValueError(f"Unrecognized time format: {time_str}") from e

    return dt.strftime("%H:%M:%S")

def prep_start_and_end_times(
    start_str: str,
    end_str: str,
    when: Optional[date] = None,
    allow_rollover: bool = True,
) -> Tuple[int, int]:
    """
    Convert 'HH:MM:SS AM/PM' strings to UTC epoch timestamps.
    If `when` is None, uses today's date (UTC).
    """
    if when is None:
        when = datetime.now(timezone.utc).date()

    def parse_hms(hms: str) -> datetime:
        t = datetime.strptime(hms, "%H:%M:%S").time()
        return datetime(when.year, when.month, when.day, t.hour, t.minute, t.second, tzinfo=timezone.utc)
    
    norm_start_str = normalize_time_str(start_str)
    norm_end_str = normalize_time_str(end_str)
    start_dt = parse_hms(norm_start_str)
    end_dt = parse_hms(norm_end_str)

    if allow_rollover and end_dt < start_dt:
        end_dt += timedelta(days=1)
    
    return int(start_dt.timestamp()), int(end_dt.timestamp())

def get_only_undeveloped_raw_files(lts_jpg_dst: Path, raw_files: list[Path]) -> list[Path]:
    undeveloped_raw_files = []
    for raw_file in raw_files:
        raw_file_stem = raw_file.stem
        developed_jpg = lts_jpg_dst / f"{raw_file_stem}.jpg"
        if not developed_jpg.exists():
            undeveloped_raw_files.append(raw_file)
    return undeveloped_raw_files
    
def get_files(cfg: DictConfig, task: str) -> List[Path]:
    """
    Retrieve files for the specified task and optional time range.
    """
    batch_id = cfg.batch_id
    date_str = batch_id.split("_")[-1]
    date_split = date_str.split("-")
    date_time = date(int(date_split[0]), int(date_split[1]), int(date_split[2]))
    local_data_dir = Path(cfg.paths.data_dir)
    if "inspect_images" in task or "move_data" in task or "report" in task or "no_remap_label" in task or "detect_plants" in task:
        lts_dir = Path(find_lts_dir(batch_id, cfg.paths.lts_locations, local=False, jpgs=True, developed=True))
    else:
        lts_dir = Path(find_lts_dir(batch_id, cfg.paths.lts_locations, local=False))
    # raw_dir = find_raw_dir(local_data_dir, batch_id, lts_dir)
    lts_jpg_dst = lts_dir / "semifield-developed-images" / batch_id / "images"

    start_time_raw = cfg.get("start_time", None)
    end_time_raw   = cfg.get("end_time", None)

    start_time = (start_time_raw or "").upper()
    end_time   = (end_time_raw or "").upper()

    if not start_time or not end_time:
        log.info("No start_time/end_time provided; using full time range (no filtering).")
    else:
        log.info(f"Filtering files between {start_time} and {end_time}")

    def _filter_by_time(files):
        if start_time and end_time:
            start_epoch, end_epoch = prep_start_and_end_times(start_time, end_time, date_time)
            log.info(f"Filtering files between epochs {start_epoch} and {end_epoch}")
            return filter_files_by_timestamp(files, start_epoch, end_epoch)
        return files

    if task == "raw2jpg":
        raw_dir = find_raw_dir(local_data_dir, batch_id, lts_dir)
        raw_files = sorted([f for mask in cfg.file_masks.raw_files for f in raw_dir.glob(f"*{mask}")])
        log.info(f"Found {len(raw_files)} RAW files.")
        raw_files = _filter_by_time(raw_files)
        log.info(f"Found {len(raw_files)} RAW files (filtered).")
        undeveloped = get_only_undeveloped_raw_files(lts_jpg_dst, raw_files)
        log.info(f"Found {len(undeveloped)} undeveloped RAW files.")
        sampled = set(random.sample(undeveloped, min(len(undeveloped), cfg.raw2jpg.jpg_samples)))
        sample_org = [(f, f in sampled) for f in raw_files]
        
        return sample_org

    elif task in {"update_exif", "report", "report_developed"}:
        images = _filter_by_time(sorted(lts_jpg_dst.glob("*.jpg")))
        return images

    elif task == "auto_sfm":
        down_photos = Path(cfg.paths.down_photos)
        photos = _filter_by_time(sorted(down_photos.glob("*.jpg")) + sorted(down_photos.glob("*.JPG")))
        return [str(p) for p in photos]

    elif task == "detect_plants":
        images = _filter_by_time(list(lts_jpg_dst.glob("*.jpg")) + list(lts_jpg_dst.glob("*.JPG")))
        return images

    elif task == "remap_labels":
        batch_dir = Path(cfg.paths.batch_dir)
        metadata_path = Path(cfg.paths.autosfm) / "reference" / f"{batch_dir.name}_metadata.csv"
        metadata = pd.read_csv(metadata_path)
        image_ids = _filter_by_time(sorted(metadata["image_id"].unique()))
        return image_ids

    elif task == "assign_species":
        metadata_files = _filter_by_time(sorted((Path(cfg.paths.batch_dir) / "metadata").glob("*.json")))
        return metadata_files

    elif task == "no_remap_label":
        csv_files = _filter_by_time(list(Path(cfg.paths.plant_detection_dir).glob("*.csv")))
        return csv_files

    elif task == "inspect_images":
        return _filter_by_time(sorted(lts_jpg_dst.glob("*.jpg")))

    elif task == "inspect_images_jsons":
        lts_meta = lts_dir / "semifield-developed-images" / batch_id / "metadata"
        local_meta = Path(cfg.paths.batch_dir) / "metadata"
        metadata_dir = local_meta if local_meta.exists() else lts_meta
        return _filter_by_time(sorted(metadata_dir.glob("*.json")))

    elif task == "move_data_temp_data":
        return _filter_by_time(sorted((Path(cfg.paths.batch_dir) / "metadata").glob("*.json")))

    elif task == "move_data_lts_data":
        lts_meta = lts_dir / "semifield-developed-images" / batch_id / "metadata"
        return _filter_by_time(sorted(lts_meta.glob("*.json")))

    elif task == "report_uploads":
        raw_dir = find_raw_dir(local_data_dir, batch_id, lts_dir)
        remote_raw_dir = lts_dir / "semifield-upload" / batch_id
        if "3.1" in str(cfg.bbot_version):
            ext = "*.RAW"
        elif (remote_raw_dir / "SONY").exists():
            ext = "SONY/*.ARW"
        else:
            ext = "*.ARW"
        return _filter_by_time(sorted(remote_raw_dir.glob(ext)))

    else:
        raise ValueError(f"Unknown task: {task}")

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
    
    lts_dir = find_lts_dir(cfg.batch_id, cfg.paths.lts_locations, developed=True, jpgs=True)
    
    fov_csv = Path(lts_dir) / "semifield-developed-images" / cfg.batch_id / "reference" / "fov.csv"

    if fov_csv.exists():
        log.info(f"Found fov.csv at {fov_csv}. Assuming batch is reconstructed.")
        return True
    
    if no_remap_label_status == "success":
        log.info("No remap label task detected. Using simple labels.")
        return False

    if remap_labels_status == "success" and no_remap_label_status != "success" and autosfm_status == "success":
        log.info("Remap label task detected. Using reconstructed labels.")
        return True
    
    return None

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
                    if cfg.season is None and "pipeline_season" in v:
                        cfg.season = v["pipeline_season"]
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
from pathlib import Path


def find_lts_dir(
    batch_id: str,
    nfs_locations: list[str],
    local: bool = False,
    developed: bool = False,
    dngs: bool = False,
    jpgs: bool = False,
) -> Path:
    """
    Find the parent NFS location containing a given batch.

    The function searches each candidate NFS location for the batch directory and
    verifies that the expected files exist before returning the matching parent
    NFS location.

    Search behavior:
    - `developed=False`: looks in `semifield-upload/<batch_id>` for RAW files
    - `developed=True, dngs=True`: checks for `dngs/`
    - `developed=True, jpgs=True`: looks in `images/` for JPG files
    - `developed=True` with neither `dngs` nor `jpgs`: looks in `pngs/` for PNG files

    Args:
        batch_id: Batch identifier to search for.
        nfs_locations: Candidate parent directories to search.
        local: If True, search under `data/<nfs_name>/...` instead of the full NFS path.
        developed: If True, search in `semifield-developed-images`; otherwise search
            in `semifield-upload`.
        dngs: When `developed=True`, check for a `dngs/` directory.
        jpgs: When `developed=True`, check for JPG files in `images/`.

    Returns:
        The matching parent NFS location as a Path.

    Raises:
        FileNotFoundError: If the batch directory is not found, or if it is found
            but does not contain the expected files.
        ValueError: If incompatible flag combinations are provided.
    """
    if dngs and jpgs:
        raise ValueError("`dngs` and `jpgs` cannot both be True.")

    searched_batch_dirs: list[Path] = []
    found_batch_dirs: list[Path] = []

    for root in map(Path, nfs_locations):
        base_root = Path("data") / root.name if local else root
        dataset_dir = "semifield-developed-images" if developed else "semifield-upload"
        batch_dir = base_root / dataset_dir / batch_id
        searched_batch_dirs.append(batch_dir)

        if not batch_dir.exists():
            log.error(f"Batch directory not found: {batch_dir}")
            continue

        found_batch_dirs.append(batch_dir)

        if developed:
            if dngs:
                dng_dir = batch_dir / "dngs"
                if dng_dir.exists():
                    log.info(f"Batch {batch_id} found in {batch_dir} with dngs directory present")
                    return root
                continue

            if jpgs:
                files = list((batch_dir / "images").glob("*.jpg"))
                files += list((batch_dir / "images").glob("*.JPG"))
                file_label = "JPG"
            else:
                files = list((batch_dir / "pngs").glob("*.png"))
                files += list((batch_dir / "pngs").glob("*.PNG"))
                file_label = "PNG"
        else:
            files = list(batch_dir.glob("*.RAW"))
            files += list(batch_dir.glob("*.raw"))
            file_label = "RAW"

        if files:
            log.info(
                f"Batch {batch_id} found in {batch_dir} with {len(files)} {file_label} files"
            )
            return root

    if not found_batch_dirs:
        error_message = (
            f"Batch {batch_id} not found in any candidate location: "
            f"{[str(p) for p in searched_batch_dirs]}"
        )
        log.error(error_message)
        raise FileNotFoundError(error_message)

    expected = (
        "dngs directory"
        if developed and dngs
        else "JPG files"
        if developed and jpgs
        else "PNG files"
        if developed
        else "RAW files"
    )

    error_message = (
        f"Batch {batch_id} was found, but no expected {expected} were present in: "
        f"{[str(p) for p in found_batch_dirs]}"
    )
    log.error(error_message)
    raise FileNotFoundError(error_message)

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
                    "start_time": cfg.start_time,
                    "assignee": user_id,  # from cfg.report.reviewers
                    "lts_developed": lts_dev_dir_name
                }
            }
        
    elif issue_type == "failure":
        trigger_payload = {
                "event_type": "failure-reported",
                "client_payload": {
                    "batch_id": batch_id,
                    "start_time": cfg.start_time,
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
