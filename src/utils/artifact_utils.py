import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from collections import OrderedDict

import pandas as pd
import yaml
from omegaconf import DictConfig

# Initialize logging
log = logging.getLogger(__name__)

def represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())

yaml.add_representer(OrderedDict, represent_ordereddict)
# For safety with SafeDumper too:
yaml.add_representer(OrderedDict, represent_ordereddict, Dumper=yaml.SafeDumper)


def get_empty_artifact(cfg: DictConfig, all_tasks: List[str]) -> OrderedDict:
    ordered_tasks = dict((task, None) for task in all_tasks)
    return OrderedDict([
        ("batch_id", cfg.batch_id),
        ("bbot_version", getattr(cfg, "bbot_version", "")),
        ("season", getattr(cfg, "season", "")),
        ("lts_upload_directory", cfg.paths.lts_upload_directory),
        ("lts_developed_directory", cfg.paths.lts_developed_directory),
        ("inspection_dir", f"{cfg.paths.lts_developed_directory}/{cfg.batch_id}/inspection"),
        ("task_durations", ordered_tasks.copy()),  # Maintain order here
        ("warning_and_errors", ordered_tasks.copy()),  # Maintain same order here
    ])

def flatten_tasks(cfg: DictConfig) -> list[str]:
    """Flatten all tasks across groups into a single list of unique task names."""
    all_tasks = []
    for group in cfg.tasks.values():
        all_tasks.extend(group)
    return all_tasks

def update_artifact_post_task(cfg: DictConfig, task_name: str) -> None:
    artifact_path = Path(cfg.paths.artifact_path)
    log_path = Path(cfg.paths.log_path)

    all_tasks = flatten_tasks(cfg)
    artifact = get_artifacts(artifact_path, cfg, all_tasks)

    # Ensure the task_durations and warning_and_errors sections exist
    if "task_durations" not in artifact or not isinstance(artifact["task_durations"], dict):
        artifact["task_durations"] = {}
    if "warning_and_errors" not in artifact or not isinstance(artifact["warning_and_errors"], dict):
        artifact["warning_and_errors"] = {}

    # Ensure this specific task key exists
    artifact["task_durations"].setdefault(task_name, None)
    artifact["warning_and_errors"].setdefault(task_name, None)

    # Set base metadata
    artifact.setdefault("batch_id", cfg.batch_id)
    artifact.setdefault("bbot_version", getattr(cfg, "bbot_version", ""))
    artifact.setdefault("season", getattr(cfg, "season", ""))

    artifact.setdefault("lts_upload_directory", cfg.paths.lts_upload_directory)
    artifact.setdefault("lts_developed_directory", cfg.paths.lts_developed_directory)
    artifact.setdefault("inspection_dir", f"{str(cfg.paths.lts_developed_directory)}/{cfg.batch_id}/inspection")

    try:
        log_lines = read_log_file(log_path)
    except FileNotFoundError:
        log.warning(f"No log file found at {log_path}")
        save_artifacts(artifact_path, artifact)
        return

    # Update duration
    timing_df = extract_module_timings(log_lines)
    for _, row in timing_df.iterrows():
        module_task = row["ScriptModule"].split(".")[-1]
        if module_task == task_name:
            artifact["task_durations"][module_task] = str(timedelta(seconds=round(row["DurationSeconds"])))

    # Update warnings/errors
    error_df = extract_error_blocks(log_lines)
    if not error_df.empty:
        for task in error_df["ScriptModule"].unique():
            task_key = task.split(".")[-1]
            if task_key == task_name:
                subset = error_df[error_df["ScriptModule"] == task]
                lines = [f"[{r.Level}] {r.LogSnippet} ({r.ScriptModule})" for _, r in subset.iterrows()]
                artifact["warning_and_errors"][task_key] = lines

    save_artifacts(artifact_path, artifact)


def read_log_file(log_path: Path) -> List[str]:
        if not log_path.exists():
            raise FileNotFoundError(f"Log file does not exist: {log_path}")
        with open(log_path, "r") as f:
            return f.readlines()
        
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

def read_artifact(artifact_path: str):
    with open(artifact_path, "r") as file:
        data = yaml.safe_load(file) or {}
    return data
        
def get_artifacts(artifact_path: Path, cfg: DictConfig, all_tasks: List[str]) -> OrderedDict:
    if artifact_path.exists():
        data = read_artifact(artifact_path)
        return OrderedDict(data)  # preserve loaded order if possible
    else:
        log.warning(f"Artifact YAML file not found at {artifact_path}. Initializing new artifact.")
        return get_empty_artifact(cfg, all_tasks)

def save_artifacts(artifact_path: str, data: Dict[str, Any]) -> None:
    """
    Save the given data to the YAML file at the specified artifact path.
    """
    artifact_path = Path(artifact_path)
    
    with open(artifact_path, "w") as file:
        yaml.dump(data, file, sort_keys=False, Dumper=yaml.SafeDumper)
    
    log.info(f"Artifacts saved to {artifact_path}")