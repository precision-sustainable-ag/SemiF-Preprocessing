"""
Main entry point for the SemiF-Preprocessing pipeline.

This script dynamically runs a set of tasks specified in a Hydra config file (`cfg.tasks`).

Each task is assumed to expose a `main()` function accessible via Hydra's `get_method`.
"""
import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

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

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running SemiF-Preprocesing pipeline.
    """
    cfg = OmegaConf.create(cfg)
    mode = cfg.mode
    log.info(f"Starting SemiF-Preprocessing pipeline in {mode} mode.")

    keys = read_yaml(cfg.paths.pipeline_keys)
    
    batch_id = cfg.batch_id
    state_id = batch_id.split("_")[0]
    user_id = getattr(cfg.report.reviewers.github, state_id, cfg.report.reviewers.github.default)
    os.environ["GITHUB_PAT"] = keys['GITHUB_PAT']

    lts_path  = Path(cfg.paths.lts_locations[-1]) / "semifield-developed-images"
    
    if mode not in TASK_REGISTRY:
        log.error(f"Task {mode} not found in task registry")
        return
    
    try:
        retry_nfs_access(lts_path, mode="read", retries=10)
        set_cpu_affinity()
        TASK_REGISTRY[mode](cfg)
    except Exception as e:
        log.exception(f"Error running {mode}")
        if cfg.create_issue:
            save_log_to_lts(cfg)
            log.info("Creating GitHub issue for mode failure.")
            # Trigger GitHub issue on failure
            create_issue(batch_id, user_id, issue_type="failure", tsk=mode, error_msg=str(e))
        
        log.info("Exiting due to task failure.")
        return
    
    log.info("All tasks completed successfully.")
    if cfg.create_issue:        
        log.info("Creating GitHub issue for successful run.")
        create_issue(batch_id, user_id, issue_type="report")

if __name__ == "__main__":
    main()