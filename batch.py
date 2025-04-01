import sys
import hydra
from omegaconf import DictConfig
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
    log.info("Starting batch processing.")
    
    batch_ids = cfg.batch_ids
    log.info(f"Processings {len(batch_ids)} batches")
    
    tasks = cfg.tasks
    log.info(f"Processing {len(tasks)} tasks that include {tasks}")
    
    for batch_id in batch_ids:
        log.info(f"Processing batch {batch_id}")
        cfg.batch_id = batch_id
        for task in tasks:
            if task == "autosfm":
                os.sched_setaffinity(0, {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31})
            log.info(f"Processing task {task}")
            try:
                task = get_method(f"{task}.main")
                task(cfg)
                log.info(f"Finished processing batch {batch_id} for task {task}.")
            except Exception as e:
                log.exception(f"Error processing batch {batch_id}")
    
    log.info("Finished batch processing.")
    

if __name__ == "__main__":
    main()
