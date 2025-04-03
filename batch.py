import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
from hydra.utils import get_method
import os

# Add the src directory to the PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent / "src"))

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for RAW to JPG conversion."""
    
    cfg = OmegaConf.create(cfg)
    log.info(f"Starting batch SemiF-Preprocessing pipeline with tasks: {', '.join(cfg.tasks)}")

    batch_ids = cfg.batch_ids
    log.info(f"Processings {len(batch_ids)} batches")
    
    tasks = cfg.tasks

    for batch_id in batch_ids:
        log.info(f"Processing batch {batch_id}")
        cfg.batch_id = batch_id
        for task in tasks:
            if task == "autosfm":
                try:
                    os.sched_setaffinity(0, set(range(2, 32)))
                    log.info("Set CPU affinity to cores 2-31")
                except AttributeError:
                    log.warning("CPU affinity setting not supported on this platform.")
                except Exception as e:
                    log.warning(f"Failed to set CPU affinity: {e}")

            try:
                log.info(f"Starting task: {task}")
                task = get_method(f"{task}.main")
                task(cfg)
                log.info(f"Task completed successfully: {task}")

            except Exception as e:
                log.exception(f"Task failed: {task} for batch {batch_id}")
                continue
    
    log.info("Finished batch processing.")
    

if __name__ == "__main__":
    main()
