import logging
import signal
import sys
import hydra
from omegaconf import DictConfig

from src.utils.artifact_utils import artifact_updater

from src.tasks.auto_sfm.config_utils import autosfm_present, create_config
from src.tasks.auto_sfm.metashape_utils import SfM
from src.tasks.auto_sfm.resize import (resize_masks,
                                     resize_photo_diretory)

# Set the logger
log = logging.getLogger(__name__)

@artifact_updater("autosfm")
def run_asfm_pipeline(cfg: DictConfig) -> None:

    def sigint_handler(signum, frame):
        print("\nPython SIGINT detected. Exiting.\n")
        sys.exit(1)

    def sigterm_handler(signum, frame):
        print("\nPython SIGTERM detected. Exiting.\n")
        sys.exit(1)

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)

    # Setup config
    cfg = create_config(cfg)

    # Check if autosfm has already been run
    if cfg.asfm.check_for_asfm:
        try:
            log.info(f"Checking for autosfm contents")
            if autosfm_present(cfg):
                log.info(
                    f"Autosfm has already been run. All contents are available. Moving to next process."
                )
                raise

        except Exception as e:
            log.exception(f"Failed to check asfm contents. Exiting")
            raise

    # Resize images and masks
    if cfg.asfm.resize_photos:
        try:
            if cfg.asfm.downscale.enabled:
                log.info(f"Resizing images")
                resize_photo_diretory(cfg)
                if cfg.asfm.use_masking:
                    log.info(f"Resizing masks")
                    resize_masks(cfg)
        except Exception as e:
            log.exception(f"Failed to downsize images. Exiting.")
            raise


    # Initialize pipeline
    log.info(f"Initializing SfM")
    pipeline = SfM(cfg)

    if cfg.asfm.recover_unaligned_only:
        try:
            log.info("Recovering unaligned cameras in existing chunk only")
            pipeline.recover_unaligned_only(chunk=0, rematch=cfg.asfm.recover_rematch)
            log.info("Recovery-only mode complete")
            return
        except Exception as e:
            log.exception("Failed recovery-only mode. Exiting")
            raise

    # Add photos to ms project
    if cfg.asfm.add_photos_and_masks:
        try:
            log.info(f"Adding photos")
            pipeline.add_photos()
            if cfg.asfm.use_masking:
                pipeline.add_masks()
        except Exception as e:
            log.exception(f"Failed to add photos. Exiting.")
            raise

    
    # Detect markers
    if cfg.asfm.detect_markers:
        try:
            log.info(f"Detecting markers")
            pipeline.detect_markers()
            pipeline.remove_low_id_markers()
        except Exception as e:
            log.exception(f"Failed to detect markers. Exiting")
            raise
        
    # Match photos
    if cfg.asfm.match:
        try:
            log.info(f"Matching photos")
            pipeline.match_photos()
        except Exception as e:
            log.exception(f"Failed to match photos. Exiting")
            raise

    # Align photos
    if cfg.asfm.align:
        try:
            log.info(f"Aligning photos")
            pipeline.align_photos(correct=True)
            # pipeline.reset_region()
        except Exception as e:
            log.exception(f"Failed to align photos. Exiting")
            raise

    

    # Import marker locations
    if cfg.asfm.import_references:
        try:
            log.info(f"Importing references")
            pipeline.import_reference()
        except Exception as e:
            log.exception(f"Failed to import reference. Exiting")
            raise


    # Optimize cameras
    if cfg.asfm.optimize_cameras:
        try:
            log.info(f"Optimizing cameras")
            pipeline.optimize_cameras()
        except Exception as e:
            log.exception(f"Failed to optimize cameras. Exiting")
            raise

    # Export data
    if cfg.asfm.export_gcp_camref_err:
        try:
            log.info(f"Exporting GCP reference")
            pipeline.export_gcp_reference()
        except Exception as e:
            log.exception(f"Failed to export GCP reference. Exiting")
            raise
        try:
            log.info(f"Exporting camera reference")
            pipeline.export_camera_reference()
        except Exception as e:
            log.exception(f"Failed to export camera reference. Exiting")
            raise

        try:
            log.info(f"Exporting error stats")
            pipeline.export_stats()
        except Exception as e:
            log.exception(f"Failed to export error stats. Exiting")
            raise

    # Electives

    # Build depth map
    if cfg.asfm.build_depth:
        try:
            if cfg.asfm.depth_map.enabled:
                log.info(f"Building depth maps")
                pipeline.build_depth_map()
        except Exception as e:
            log.exception(f"Failed to build depth maps. Exiting")
            raise

    # Build dense point cloud
    if cfg.asfm.build_dense:
        try:
            if cfg.asfm.dense_cloud.enabled:
                log.info(f"Buidling dense point cloud")
                pipeline.build_dense_cloud()
        except Exception as e:
            log.exception(f"Failed to buidl dense point cloud. Exiting")
            raise

    if cfg.asfm.build_model:
        try:
            if cfg.asfm.model.enabled:
                log.info(f"Buidling model")
                pipeline.build_model()
        except Exception as e:
            log.exception(f"Failed to model. Exiting")
            raise

    # Build DEM
    if cfg.asfm.build_dem:
        try:
            if cfg.asfm.dem.enabled:
                log.info(f"Building DEM")
                pipeline.build_dem()
        except Exception as e:
            log.exception(f"Failed to build DEM. Exiting")
            raise

    # Build ortho
    if cfg.asfm.build_ortho:
        try:
            if cfg.asfm.orthomosaic.enabled:
                log.info(f"Building orthomosaic")
                pipeline.build_ortomosaic()
        except Exception as e:
            log.exception(f"Failed to build orthomosaic. Exiting")
            raise

    # Export fov data
    if cfg.asfm.export_fov:
        try:
            if cfg.asfm.camera_fov.enabled:
                log.info(f"Exporting camera FOV information")
                pipeline.camera_fov()
        except Exception as e:
            log.exception(f"Failed to export camera FOV information. Exiting")
            raise

        # Export pixel-world grid
    if cfg.asfm.export_pixel_grid:
        try:
            if cfg.asfm.pixel_grid.enabled:
                log.info(f"Exporting pixel-world grids")
                pipeline.export_pixel_world_grid(step=cfg.asfm.pixel_grid.step)
        except Exception as e:
            log.exception(f"Failed to export pixel-world grids. Exiting")
            raise

    # Export preview view of ortho
    if cfg.asfm.export_report:
        try:
            log.info(f"Exporting report")
            pipeline.export_report()
        except Exception as e:
            log.exception(f"Failed to export report. Exiting")
            raise
    log.info(f"AutoSfM Complete")
    return

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main entry point for the application """
    log.info(f"Starting AutoSfM pipeline...")

    try:
        run_asfm_pipeline(cfg)
    except Exception as e:
        log.exception(f"Error running AutoSfM pipeline: {e}")
        raise

    log.info("AutoSfM pipeline complete.")
    return

if __name__ == "__main__":
    main()