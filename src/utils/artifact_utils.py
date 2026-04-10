import logging
import re
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from omegaconf import DictConfig, ListConfig

from src.utils.utils import find_lts_dir

# Initialize logging
log = logging.getLogger(__name__)

def represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())

yaml.add_representer(OrderedDict, represent_ordereddict)
# For safety with SafeDumper too:
yaml.add_representer(OrderedDict, represent_ordereddict, Dumper=yaml.SafeDumper)


def artifact_updater(task_name: str):
    def decorator(func):
        def wrapper(cfg: DictConfig, *args, **kwargs):
            # Mark as "started" BEFORE the task runs
            try:
                update_artifact_post_task(cfg, task_name, status="started")
            except Exception as e:
                log.warning(f"[artifact_updater] Failed to mark '{task_name}' as started: {e}")

            try:
                result = func(cfg, *args, **kwargs)
                update_artifact_post_task(cfg, task_name, status="success")
                return result
            except Exception as e:
                update_artifact_post_task(cfg, task_name, status="fail")
                raise
        return wrapper
    return decorator

def get_empty_artifact(cfg: DictConfig, all_tasks: List[str]) -> OrderedDict:
    default_none = {task: None for task in all_tasks}
    if cfg.paths.lts_developed_directory is None:
        try:
            lts_dir = find_lts_dir(cfg.batch_id, cfg.paths.lts_locations, developed=True, jpgs=True)
            cfg.paths.lts_developed_directory = str(Path(lts_dir) / "semifield-developed-images")
        except Exception as e:
            log.error(f"Error finding LTS developed-images directory: {e}")
    if cfg.paths.lts_upload_directory is None:
        try:
            lts_dir = find_lts_dir(cfg.batch_id, cfg.paths.lts_locations)
            cfg.paths.lts_upload_directory = str(Path(lts_dir) / "semifield-upload")
        except Exception as e:
            log.error(f"Error finding LTS upload directory: {e}")
    return OrderedDict([
        ("batch_id", cfg.batch_id),
        ("bbot_version", getattr(cfg, "bbot_version", "")),
        ("season", getattr(cfg, "season", "")),
        ("lts_upload_directory", cfg.paths.lts_upload_directory),
        ("lts_developed_directory", cfg.paths.lts_developed_directory),
        ("task_durations", default_none.copy()),
        ("warning_and_errors", default_none.copy()),
        ("task_status", default_none.copy()),
        # Optional extensions:
        # ("task_outputs", default_none.copy()),
    ])


def _set_metadata_defaults(cfg: DictConfig, artifact: Dict[str, Any]):
    artifact.setdefault("batch_id", cfg.batch_id)
    artifact.setdefault("bbot_version", getattr(cfg, "bbot_version", ""))
    artifact.setdefault("season", getattr(cfg, "season", ""))


    if artifact["lts_developed_directory"] is None:
        try:
            lts_dir = find_lts_dir(cfg.batch_id, cfg.paths.lts_locations, developed=True, jpgs=True)
            cfg.paths.lts_developed_directory = str(Path(lts_dir) / "semifield-developed-images")
            artifact["lts_developed_directory"] = str(cfg.paths.lts_developed_directory)
            artifact["inspection_dir"] = f"{str(cfg.paths.lts_developed_directory)}/{cfg.batch_id}/inspection"
        except Exception as e:
            log.error(f"Error finding LTS developed-images directory: {e}")

    if artifact["lts_upload_directory"] is None:
        try:
            lts_dir = find_lts_dir(cfg.batch_id, cfg.paths.lts_locations)
            cfg.paths.lts_upload_directory = Path(lts_dir) / "semifield-upload"
            artifact.setdefault("lts_upload_directory", cfg.paths.lts_upload_directory)
        except Exception as e:
            log.error(f"Error finding LTS upload directory: {e}")

def _update_duration_from_logs(artifact: Dict[str, Any], log_path: Path, task_name: str) -> None:
    """
    Parse the log file and update artifact["task_durations"][task_name] with time delta.
    """
    try:
        log_lines = read_log_file(log_path)
    except FileNotFoundError:
        log.warning(f"No log file found at {log_path}")
        return

    timing_df = extract_module_timings(log_lines)
    for _, row in timing_df.iterrows():
        module_task = row["ScriptModule"].split(".")[-1]
        if module_task == task_name:
            duration = str(timedelta(seconds=round(row["DurationSeconds"])))
            artifact["task_durations"][task_name] = duration

def _update_warnings_from_logs(artifact: Dict[str, Any], log_path: Path, task_name: str) -> None:
    """
    Parse the log file and update artifact["warning_and_errors"][task_name] with any relevant log messages.
    """
    try:
        log_lines = read_log_file(log_path)
    except FileNotFoundError:
        log.warning(f"No log file found at {log_path}")
        artifact["warning_and_errors"][task_name] = None
        return

    error_df = extract_error_blocks(log_lines)
    if not error_df.empty:
        subset = error_df[error_df["ScriptModule"].str.endswith(task_name)]
        if not subset.empty:
            lines = [f"[{r.Level}] {r.LogSnippet} ({r.ScriptModule})" for _, r in subset.iterrows()]
            artifact["warning_and_errors"][task_name] = lines
        else:
            artifact["warning_and_errors"][task_name] = None
    else:
        artifact["warning_and_errors"][task_name] = None

def update_artifact_post_task(cfg: DictConfig, task_name: str, status: str = "success") -> None:
    artifact_path = Path(cfg.paths.artifact_path)
    log_path = Path(cfg.paths.log_path)
    all_tasks = flatten_tasks(cfg)
    artifact = get_artifacts(artifact_path, cfg, all_tasks)

    _set_metadata_defaults(cfg, artifact)

    # Ensure necessary fields are initialized
    artifact.setdefault("task_durations", {})
    artifact.setdefault("warning_and_errors", {})
    artifact["task_durations"].setdefault(task_name, None)
    artifact["warning_and_errors"].setdefault(task_name, None)

    if "task_status" not in artifact or not isinstance(artifact["task_status"], dict):
        artifact["task_status"] = {}
    artifact["task_status"].setdefault(task_name, None)

    # Update from logs
    _update_duration_from_logs(artifact, log_path, task_name)
    _update_warnings_from_logs(artifact, log_path, task_name)
    
    artifact["task_status"][task_name] = status  # ✅ write status

    save_artifacts(artifact_path, artifact)
        
def extract_error_blocks(log_lines: List[str]) -> pd.DataFrame:
    """
    Extracts blocks of log entries beginning with ERROR or WARNING.

    Returns:
        pd.DataFrame: DataFrame containing index, module name, and error/warning log blocks.
    """
    messages = []
    modules = []
    levels = []

    for line in log_lines:
        match = re.search(r"\[\d{4}-\d{2}-\d{2}.*?\]\[([^\]]+)\]\[(ERROR|WARNING)\]\s+-\s+(.*)", line)
        if match:
            module = match.group(1)
            level = match.group(2)
            message = match.group(3).strip()
            modules.append(module)
            levels.append(level)
            messages.append(message)

    return pd.DataFrame({
        "ErrorIndex": range(1, len(messages) + 1),
        "ScriptModule": modules,
        "Level": levels,
        "LogSnippet": messages
    })

def extract_module_timings(log_lines: List[str]) -> pd.DataFrame:
    """
    Calculates total active time for each logical script module using first and last timestamps.
    Merges entries that map to the same logical module and calculates total active time and boundaries.
    
    Returns:
        pd.DataFrame: DataFrame with ScriptModule, StartTime, EndTime, DurationSeconds.
    """
    module_times = defaultdict(list)

    for line in log_lines:
        match = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\]\[([^\]]+)\]", line)
        if match:
            timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            module = match.group(2)
            module_times[module].append(timestamp)

    records = []

    all_timestamps = []

    for module, times in module_times.items():
        times.sort()
        all_timestamps.extend(times)
        duration = (times[-1] - times[0]).total_seconds()
        records.append({
            "ScriptModule": module,
            "StartTime": times[0],
            "EndTime": times[-1],
            "DurationSeconds": duration
        })

    # Add total duration across all modules
    first_timestamp = min(all_timestamps)
    last_timestamp = max(all_timestamps)
    total_duration = (last_timestamp - first_timestamp).total_seconds()
    records.append({
        "ScriptModule": "Total",
        "StartTime": first_timestamp,
        "EndTime": last_timestamp,
        "DurationSeconds": total_duration
    })

    df = pd.DataFrame(records)
    df = df[df["ScriptModule"] != "pyogrio._io"]
    df = df[df["ScriptModule"] != "__main__"]
    return df

def flatten_tasks(cfg: DictConfig) -> List[str]:
    """Return a flat list of all task names from all groups."""
    tasks = []

    for group_name, group in cfg.tasks.items():
        if group is None:
            continue

        if not isinstance(group, (list, ListConfig)):
            raise TypeError(f"Task group '{group_name}' must be a list, got {type(group)}")

        tasks.extend(group)

    return tasks

def read_log_file(log_path: Path) -> List[str]:
        if not log_path.exists():
            raise FileNotFoundError(f"Log file does not exist: {log_path}")
        with open(log_path, "r") as f:
            return f.readlines()
        
def read_artifact(artifact_path: str):
    with open(artifact_path, "r") as file:
        return yaml.safe_load(file) or {}
        
def get_artifacts(artifact_path: Path, cfg: DictConfig, all_tasks: List[str]) -> OrderedDict:
    if artifact_path.exists():
        return OrderedDict(read_artifact(artifact_path))
    log.warning(f"Artifact YAML file not found at {artifact_path}. Initializing new artifact.")
    return get_empty_artifact(cfg, all_tasks)

def save_artifacts(artifact_path: str, data: Dict[str, Any]) -> None:
    """
    Save the given data to the YAML file at the specified artifact path.
    """
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(artifact_path, "w") as file:
        yaml.dump(data, file, sort_keys=False, Dumper=yaml.SafeDumper)
    
    log.info(f"Artifacts saved to {artifact_path}")