import sys
from pathlib import Path
import logging
import os

# Add the src directory to the PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_method

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running SemiF-Preprocesing pipeline.
    """
    cfg = OmegaConf.create(cfg)
    log.info(f"Starting task {','.join(cfg.tasks)}")
    
    for tsk in cfg.tasks:
        if tsk == "autosfm":
            os.sched_setaffinity(0, {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31})
        try:
            task = get_method(f"{tsk}.main")
            task(cfg)

        except Exception as e:
            log.exception("Failed")
            return


if __name__ == "__main__":
    main()