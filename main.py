import logging
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf  # Do not confuse with dataclass.MISSING

# Import the task functions
from src.copy_from_lockers import main as copy_from_lockers


log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "copy_from_lockers": copy_from_lockers,
    # Add more tasks here as needed
}


log = logging.getLogger(__name__)
# Get the logger for the Azure SDK

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    log.info(f"Starting task {','.join(cfg.tasks)}")
    
    for tsk in cfg.tasks:
        try:            
            TASK_REGISTRY[tsk](cfg)

        except Exception as e:
            log.exception("Failed")
            return


if __name__ == "__main__":
    main()