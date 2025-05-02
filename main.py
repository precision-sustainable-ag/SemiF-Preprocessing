"""
Main entry point for the SemiF-Preprocessing pipeline.

This script dynamically runs a set of tasks specified in a Hydra config file (`cfg.tasks`).

Each task is assumed to expose a `main()` function accessible via Hydra's `get_method`.
"""
import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path

# Ensure 'src/' is in the Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "tasks" / "label_utils"))

from src.autosfm import main as asfm
from src.correct import main as correct
from src.deliver import main as deliver
from src.label import main as label
from src.sync_from_remote import main as sync
from src.utils.utils import (
    create_issue,
    read_yaml,
    retry_nfs_access,
    save_log_to_lts,
    set_cpu_affinity,
)

# Define a registry of tasks
TASK_REGISTRY = {
    "sync": sync,
    "correct": correct,
    "asfm": asfm,
    "label": label,
    "deliver": deliver,
    # Add more tasks here as needed
}

# Set up global logger with the standardized format
log = logging.getLogger(__name__)

def run_single_batch(cfg: DictConfig, batch_cfg: dict = None) -> None:
    if batch_cfg:
        cfg.batch_id = batch_cfg["batch_id"]
        cfg.bbot_version = batch_cfg["bbot_version"]
        cfg.season = batch_cfg["season"]
    
    modes = cfg.modes
    log.info(f"Running pipeline for batch {cfg.batch_id} in {','.join(modes)} mode.")

    keys = read_yaml(cfg.paths.pipeline_keys)
    state_id = cfg.batch_id.split("_")[0]
    user_id = getattr(cfg.report.reviewers.github, state_id, cfg.report.reviewers.github.default)
    os.environ["GITHUB_PAT"] = keys['GITHUB_PAT']

    lts_path = Path(cfg.paths.lts_locations[-1]) / "semifield-developed-images"
    
    for mode in modes:
        if mode not in TASK_REGISTRY:
            log.error(f"Task {mode} not found in task registry")
            raise ValueError(f"Task {mode} not found in task registry")
        try:
            retry_nfs_access(lts_path, mode="read", retries=10)
            set_cpu_affinity()
            TASK_REGISTRY[mode](cfg)
        except Exception as e:
            log.exception(f"Error running {mode}")
            if cfg.create_issue:
                save_log_to_lts(cfg)
                log.info("Creating GitHub issue for mode failure.")
                create_issue(cfg.batch_id, user_id, issue_type="failure", tsk=mode, error_msg=str(e))
            log.info("Exiting due to task failure.")
            raise

    log.info(f"Finished batch {cfg.batch_id} successfully.")
    if cfg.create_issue:
        log.info("Creating GitHub issue for successful run.")
        create_issue(cfg.batch_id, user_id, issue_type="report")
    
    return 


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.run_mode == "batch":

        if "batch_list" not in cfg or not cfg.batch_list:
            log.error("Batch mode specified but batch_list is missing in config.")
            raise ValueError("Missing batch_list for batch mode.")
        
        for batch_cfg in cfg.batch_list:
            try:
                run_single_batch(cfg, batch_cfg)
            except Exception as e:
                log.exception(f"Error processing batch {batch_cfg['batch_id']}: {e}")
                continue
    else:
        run_single_batch(cfg)

if __name__ == "__main__":
    main()