import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def read_config(cfg: DictConfig):
    # Print batch IDs as a space-separated string for shell script
    print(" ".join(cfg.preprocess_batches.batch_ids))

if __name__ == "__main__":
    read_config()
