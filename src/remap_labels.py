import logging
import time
from pathlib import Path

from bbox.bbox_transformations import BBoxFilter, BBoxMapper
from bbox.connectors import BBoxComponents, SfMComponents
from bbox.io_utils import ParseYOLOCsv
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class RemapLabels:
    def __init__(self, cfg: DictConfig) -> None:
        self.batch_id = cfg.batch_id
        self.data_dir = Path(cfg.paths.data_dir)
        self.developed_dir = Path(cfg.paths.developed_dir)  # data_root
        self.batch_dir = Path(cfg.paths.batch_dir)
        self.batch_id = self.batch_dir.name
        self.autosfmdir = Path(cfg.paths.autosfm)
        self.metadata = self.batch_dir / "metadata"
        self.reference = self.autosfmdir / "reference"
        self.fov_cams = Path(self.autosfmdir, "reference", "fov.csv")
        self.raw_label = self.autosfmdir / "detections.csv"
        self.downscaled = cfg.asfm.downscale.enabled
        self.state_id = self.batch_id.split("_")[0]

        if self.downscaled:
            self.image_dir = self.autosfmdir / "downscaled_photos"
        else:
            self.image_dir = self.batch_dir / "images"
        self.fullres_image_path = self.batch_dir / "images"

    @property
    def camera_reference(self):
        connector = SfMComponents(self.reference)
        gcp_reference = connector.gcp_reference
        cam_ref = connector.camera_reference
        return cam_ref

    def remap_labels(self):
        # Initialize the reader which will read the annotation files and convert the bounding
        # boxes to the desired format
        reader = ParseYOLOCsv(
            image_path=self.image_dir,
            label_path=self.raw_label,
            fullres_image_path=self.fullres_image_path,
            fov_cams=self.fov_cams,
        )

        # Initialize the connector and get a list of all the images
        box_connector = BBoxComponents(
            self.data_dir,
            self.developed_dir,
            self.batch_dir,
            self.image_dir,
            self.camera_reference,
            reader,
            True,
            self.raw_label,
            fullres_image_path=self.fullres_image_path,
        )
        log.info("Fetching image metadata.")
        imgs = box_connector.images
        # Map the bounding boxes from local coordinates to global coordinate system
        log.info("Starting mapping.")
        metashape_project_path = Path(
            self.batch_dir, "autosfm", "project", f"{self.batch_id}.psx"
        )
        mapper = BBoxMapper(metashape_project_path, imgs)
        mapper.map()
        # Sanity check
        for img in imgs:
            for box in img.bboxes:
                try:
                    assert len(box._overlapping_bboxes) == 0
                except AssertionError as e:
                    log.debug("Mapping failed> Reason: {}".format(str(e)))
                    raise e
        log.info("Mapping complete.")
        # Apply "Non-Maxima Supression" to identify ideal boundng boxes.
        # Note that the operations are done in place, i.e. imgs, which is
        # a list of Image objects have BBox objects associated with them.
        # The properties of these objects are changed by the BBoxFilter.
        log.info("Deduplicating bounding boxes.")
        bbox_filter = BBoxFilter(imgs)
        bbox_filter.deduplicate_bboxes()
        log.info("Deduplication complete.")
        log.info("Saving bounding box metadata.")
        # Save the config
        for img in imgs:
            Path(self.metadata).mkdir(parents=True, exist_ok=True)
            img.image_path = Path("images", Path(img.image_path).name)
            img.save_config(self.metadata)
        log.info("Saving complete.")
        return imgs


def main(cfg: DictConfig) -> None:
    start = time.time()
    rmpl = RemapLabels(cfg)
    imgs = rmpl.remap_labels()
    end = time.time()
    log.info(f"Remap completed in {end - start} seconds.")
