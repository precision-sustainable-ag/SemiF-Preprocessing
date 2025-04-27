"""
Main entry point for the SemiF-Preprocessing pipeline.

This script dynamically runs a set of tasks specified in a Hydra config file (`cfg.tasks`).

Each task is assumed to expose a `main()` function accessible via Hydra's `get_method`.
"""

import sys
from pathlib import Path
import logging
import os
# Add the src directory to the PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_method

from utils.utils import read_yaml, save_log_to_lts, create_issue, retry_nfs_access

# Set up global logger with the standardized format
log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running SemiF-Preprocesing pipeline.
    """
    cfg = OmegaConf.create(cfg)
    log.info(f"Starting SemiF-Preprocessing pipeline with tasks: {', '.join(cfg.tasks)}")

    keys = read_yaml(cfg.paths.pipeline_keys)
    batch_id = cfg.batch_id
    state_id = batch_id.split("_")[0]
    user_id = getattr(cfg.report.reviewers.github, state_id, cfg.report.reviewers.github.default)
    os.environ["GITHUB_PAT"] = keys['GITHUB_PAT']

    lts_path  = Path(cfg.paths.lts_locations[-1]) / "semifield-developed-images"
    
    for tsk in cfg.tasks:
        retry_nfs_access(lts_path, mode="read", retries=10)
        # Optional CPU affinity settings for performance tuning on specific tasks
        if tsk == "autosfm":
            try:
                os.sched_setaffinity(0, set(range(2, 32)))
                log.info("Set CPU affinity to cores 2-31")
            except AttributeError:
                log.warning("CPU affinity setting not supported on this platform.")
            except Exception as e:
                log.warning(f"Failed to set CPU affinity: {e}")
        try:
            # Dynamically load and run task module's main function
            log.info(f"Starting task: {tsk}")
            task = get_method(f"{tsk}.main")
            task(cfg)
            log.info(f"Task completed successfully: {tsk}")

        except Exception as e:
            log.exception(f"Task failed: {tsk}")
            log.error(f"Error details: {e}")

            if cfg.create_issue:
                save_log_to_lts(cfg)
                log.info("Creating GitHub issue for task failure.")
                # Trigger GitHub issue on failure
                create_issue(batch_id, user_id, issue_type="failure", tsk=tsk, error_msg=str(e))

            log.info("Exiting due to task failure.")
            sys.exit(1)
    
    log.info("All tasks completed successfully.")
    if cfg.create_issue:        
        log.info("Creating GitHub issue for successful run.")
        create_issue(batch_id, user_id, issue_type="report")

if __name__ == "__main__":
    main()