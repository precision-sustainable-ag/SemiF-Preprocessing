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

from utils.utils import read_yaml

import subprocess
import json

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
    state_id = cfg.batch_id.split("_")[0]
    user_id = getattr(cfg.report.reviewers.github, state_id, cfg.report.reviewers.github.default)
    globus_link_prefix = keys['globus_link']
    os.environ["GITHUB_PAT"] = keys['GITHUB_PAT']

    
    for tsk in cfg.tasks:
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
            log.exception(f"Task failed: {tsk}")  # Exception includes traceback
            log.error(f"Error details: {e}")
            
            # Placeholder for Slack notification logic
            log.info("Exiting due to task failure.")
            
            if cfg.slack_report:
                message = f"Task {tsk} failed for {cfg.batch_id}"

                if tsk != "autosfm":
                    globus_pdf_link = f"{globus_link_prefix}/{cfg.batch_id}/inspection/{cfg.batch_id}_asfm_report.pdf"
                    message = f"{message}\nGo to this link to ASfM results: {globus_pdf_link}"
                
                # summ_mesage = generate_summary_message(message, user_id=user_id, message_type="Error")
                # send_slack_notification(cfg, summ_mesage, files=[])
        
            sys.exit(1)
    
    log.info("All tasks completed successfully.")
    if cfg.slack_report:        
        message = f"All tasks completed successfully for {cfg.batch_id}"
        
        if "report" in cfg.tasks:
            # globus_pdf_link = f"{globus_link_prefix}/{cfg.batch_id}/inspection/{cfg.batch_id}_report.pdf"
            # message = f"{message}\n\nInspect results here and file any issues in the SemiF-Preprocessing repo: {globus_pdf_link}"

            trigger_payload = {
                "event_type": "report-generated",
                "client_payload": {
                    "batch_id": cfg.batch_id,
                    "assignee": user_id  # from cfg.report.reviewers
                }
            }

            subprocess.run([
                "curl", "-X", "POST", "https://api.github.com/repos/precision-sustainable-ag/SemiF-Preprocessing/dispatches",
                "-H", f"Authorization: token {os.environ['GITHUB_PAT']}",
                "-H", "Accept: application/vnd.github.v3+json",
                "-d", json.dumps(trigger_payload)
            ], check=True)


        # summ_message = generate_summary_message(message, user_id=user_id, message_type="Info")
        # send_slack_notification(cfg, summ_message, files=[])


if __name__ == "__main__":
    main()