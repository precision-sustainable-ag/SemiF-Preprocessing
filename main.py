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
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_method

from utils.slack_message import generate_summary_message, send_slack_notification

# Set up global logger with the standardized format
log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running SemiF-Preprocesing pipeline.
    """
    cfg = OmegaConf.create(cfg)
    log.info(f"Starting SemiF-Preprocessing pipeline with tasks: {', '.join(cfg.tasks)}")
    
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
                message = generate_summary_message(f"Task {tsk} failed for {cfg.batch_id}", message_type="Error")
                log_file = Path(HydraConfig.get().runtime.output_dir) / f"{cfg.batch_id}.log"
                if tsk != "autosfm":
                    final_report_file = Path(cfg.paths.inspection_dir) / f"{cfg.batch_id}_asfm_report.pdf"
                    files = [x for x in [log_file, final_report_file] if x.exists()] # add ASFM report if it exists to provide more context
                else:
                    files = [log_file]
                
                send_slack_notification(cfg, message, files=[files])
            sys.exit(1)
    
    log.info("All tasks completed successfully.")
    if cfg.slack_report and "report" in cfg.tasks:
        message = generate_summary_message(f"All tasks completed successfully for {cfg.batch_id}", message_type="Info")
        final_report_file = Path(cfg.paths.inspection_dir) / f"{cfg.batch_id}_report.pdf"
        send_slack_notification(cfg, message, files=[final_report_file])


if __name__ == "__main__":
    main()