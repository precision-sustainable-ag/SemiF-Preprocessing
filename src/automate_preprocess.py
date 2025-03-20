import hydra
from omegaconf import DictConfig
import logging
from src.raw2jpg import Raw2Jpg

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for RAW to JPG conversion."""
    log.info("Starting batch processing of raw to jpg conversion")
    batch_ids = cfg.batch_ids
    log.info(f"Processings {len(batch_ids)} batches")
    for batch_id in batch_ids:
        log.info(f"Processing batch {batch_id}")
        cfg.batch_id = batch_id
        try:
            converter = Raw2Jpg(cfg)
            converter.process_files()
            log.info(f"Finished processing batch {batch_id}")
        except Exception as e:
            log.exception(f"Error processing batch {batch_id}")
    
    log.info("Finished batch processing of raw to jpg conversion")
    

if __name__ == "__main__":
    main()
