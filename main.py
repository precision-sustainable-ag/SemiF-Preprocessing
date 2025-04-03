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
            continue  # Continue running other tasks instead of exiting early


if __name__ == "__main__":
    main()