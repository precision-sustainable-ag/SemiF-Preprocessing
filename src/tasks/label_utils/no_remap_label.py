import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional , Tuple

import pandas as pd
from omegaconf import DictConfig
import hydra
from src.utils.utils import get_files
from src.utils.datasets import (
    CameraCoefficients,
    FOV,
    CameraInfo,
    BoundingBox,
    ImageMetadata,
    BBoxCoordinates,
    GlobalCoordinates,
)
from src.tasks.label_utils.bbox_area_estimation import (
    estimate_bbox_area_sqm,
    sensor_pixel_pitch_mm,
)

log = logging.getLogger(__name__)

class ImageMeta:
    """
    Represents all metadata for a single image, including camera information and bounding box annotations.

    Attributes:
        image_id (str): Unique identifier for the image.
        season (str): collection season.
        batch_id (str): Batch identifier.
        bbot_version (str): Version of the imaging platform.
        fullres_width (int): Full-resolution image width (pixels).
        fullres_height (int): Full-resolution image height (pixels).
        camera_info (CameraInfo): Camera information and calibration fields.
        exif_meta: EXIF metadata for the image (optional/extendable).
        annotations (List[BoundingBox]): List of bounding box annotations.
        datetime (str): ISO-format string for image capture datetime (derived from image_id).
    """
    def __init__(self, image_id: str, season: str, batch_id: str, bbot_version: str,
                 fullres_width: int, fullres_height: int, camera_info: CameraInfo):
        self.image_id = image_id
        self.season = season
        self.batch_id = batch_id
        self.bbot_version = bbot_version
        self.fullres_width = fullres_width
        self.fullres_height = fullres_height
        self.camera_info = camera_info
        self.exif_meta = None  # Extend as needed
        self.annotations: List[BoundingBox] = []
        self.datetime = self._get_datetime_from_id(image_id)

    @staticmethod
    def _get_datetime_from_id(image_id: str) -> str:
        """
        Extracts a datetime string from the image_id, falling back to a prefix if not epoch-encoded.
        Returns:
            str: ISO 8601 formatted datetime string or fallback value.
        """
        try:
            epoch_time = int(image_id.split("_")[1])
            return pd.to_datetime(epoch_time, unit="s").isoformat()
        except Exception:
            return image_id.split("_")[0]

    def add_annotation(self, annotation: BoundingBox) -> None:
        # Appends a BoundingBox annotation to this image's annotation list.
        self.annotations.append(annotation)

    def to_dataclass(self) -> ImageMetadata:
        # Converts this ImageMeta instance into a serializable ImageMetadata dataclass.
        return ImageMetadata(
            season=self.season,
            datetime=self.datetime,
            bbot_version=self.bbot_version,
            image_id=self.image_id,
            batch_id=self.batch_id,
            validated=False,
            version="1",
            exif_meta=self.exif_meta,
            camera_info=self.camera_info,
            fullres_width=self.fullres_width,
            fullres_height=self.fullres_height,
            annotations=self.annotations,
        )

class RemapLabelsPipeline:
    """
    Main pipeline for remapping detection outputs to structured metadata JSON files.

    Responsible for loading detection results, mapping class names to IDs, creating metadata objects,
    and saving structured metadata JSONs for each image.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Attributes:
        season (str): collection season.
        batch_id (str): Identifier for batch of images.
        bbot_version (str): Imaging hardware version.
        detections_dir (Path): Directory with detection CSVs.
        output_dir (Path): Output directory for JSON metadata.
        species_info_remapped (Dict): Map of common names to species info dicts.
        fullres_width (int): Image width in pixels.
        fullres_height (int): Image height in pixels.
        species_class (str): The class name assigned to all bboxes (manual override).
        df (pd.DataFrame): Merged detection data for the batch.
    """
    def __init__(self, cfg: DictConfig):
        # Sanity check
        self._check_assign_species(cfg)
        
        self.cfg = cfg
        self.season = cfg.season
        self.batch_id = cfg.batch_id
        self.bbot_version = str(cfg.bbot_version)
        self.z_axis = float(cfg.z_axis)
        self.cam_angle = float(cfg.cam_angle)

        self.output_dir = self._init_output_dir(cfg)
        self.species_info_remapped = self._remap_species_info(cfg.paths.species_info)
        
        self.fullres_width, self.fullres_height = self._determine_fullres_dims(cfg, self.bbot_version)
        self.sensor_width_mm, self.sensor_height_mm = self._determine_sensor_dims(cfg)
        self.focal_length_mm = float(cfg.exif.FocalLength)
        self.pixel_width_mm, self.pixel_height_mm = sensor_pixel_pitch_mm(
            self.sensor_width_mm,
            self.sensor_height_mm,
            self.fullres_width,
            self.fullres_height,
        )
        
        self.species_class = cfg.assign_species.assign_all_bboxes.label.lower()
        self.df = self._load_detections()

    @staticmethod
    def _init_output_dir(cfg: DictConfig) -> Path:
        # Returns and creates (if necessary) the output directory for metadata JSONs.
        output_dir = Path(cfg.paths.batch_dir) / "metadata"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    @staticmethod
    def _remap_species_info(species_info_json: str) -> Dict[str, Any]:
        """
        Loads the species info JSON and remaps it from code to common name for class lookup.
        Args:
            species_info_json (str): Path to the species info JSON file.
        Returns:
            Dict[str, Any]: Map of common names to species metadata dicts.
        """
        with open(species_info_json, "r") as f:
            species_info = json.load(f)
        return {v["common_name"].lower(): v for _, v in species_info["species"].items()}

    @staticmethod
    def _determine_fullres_dims(cfg: DictConfig, bbot_version: str) -> Tuple[int, int]:
        # Returns image size based on camera version.
        if "3.1" in bbot_version:
            return cfg.exif.SVCamImageWidth, cfg.exif.SVCamImageHeight
        else:
            return cfg.exif.SonyImageWidth, cfg.exif.SonyImageHeight

    @staticmethod
    def _determine_sensor_dims(cfg: DictConfig) -> Tuple[float, float]:
        return float(cfg.exif.SensorWidth), float(cfg.exif.SensorHeight)

    @staticmethod
    def _check_assign_species(cfg: DictConfig) -> None:
        # Checks that manual species assignment is enabled in config, raises if not.
        if not cfg.assign_species.assign_all_bboxes.enabled:
            raise ValueError("Manual species assignment is not enabled.")

    def _load_detections(self) -> pd.DataFrame:
        csv_files = get_files(self.cfg, task="no_remap_label")
        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            df["image_id"] = f.stem
            df["z_axis"] = self.z_axis
            df["cam_angle"] = self.cam_angle
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def _class_id(self, class_name: str) -> int:
        """
        Gets the integer class ID for a given class name.

        Args:
            class_name (str): Detected object class name.

        Returns:
            int: Integer class ID for metadata.
        """

        if class_name == "plant":
            return self.species_info_remapped[self.species_class]["class_id"]
        elif class_name == "colorchecker":
            return self.species_info_remapped["colorchecker"]["class_id"]
        else:
            return self.species_info_remapped["unknown"]["class_id"]

    def _bbox_xywh(self, row: pd.Series, fullres_width: int, fullres_height: int) -> Optional[List[int]]:

        """
        Converts normalized detection bbox [xmin, ymin, xmax, ymax] to full res pixel [x, y, w, h].
        Returns None if any bbox coordinate is NaN.

        Args:
            row (pd.Series): Detection row.
            fullres_width (int): Image width in pixels.
            fullres_height (int): Image height in pixels.

        Returns:
            Optional[List[int]]: Pixel bounding box or None if data invalid.
        """

        if pd.isna(row["xmin"]) or pd.isna(row["ymin"]) or pd.isna(row["xmax"]) or pd.isna(row["ymax"]):
            log.warning(f"Skipping bbox conversion due to NaN values: {row}")
            return None
        x = round(row["xmin"] * fullres_width)
        y = round(row["ymin"] * fullres_height)
        w = round((row["xmax"] * fullres_width)) - x
        h = round((row["ymax"] * fullres_height)) - y
        return [x, y, w, h]

    def _estimate_bbox_area_sqm(self, bbox_xywh: Optional[List[int]]) -> Optional[float]:
        return estimate_bbox_area_sqm(
            bbox_xywh=bbox_xywh,
            pixel_width_mm=self.pixel_width_mm,
            pixel_height_mm=self.pixel_height_mm,
            focal_length_mm=self.focal_length_mm,
            z_axis_cm=self.z_axis,
            cam_angle_deg=self.cam_angle,
        )
    
    def process(self):

        """
        Runs the entire pipeline: processes detections, builds metadata, and writes JSON outputs.
        For each image, creates an ImageMeta, adds annotations, and saves the metadata.
        """
        # Iterate through each image_id group in the DataFrame
        for image_id, group in self.df.groupby("image_id"):
            
            # Create ImageMeta object for the current image
            try:
                image_meta = ImageMeta(
                    image_id=image_id,
                    season=self.season,
                    batch_id=self.batch_id,
                    bbot_version=self.bbot_version,
                    fullres_width=self.fullres_width,
                    fullres_height=self.fullres_height,
                    camera_info=CameraInfo(
                        fov=FOV(),
                        cam_angle=self.cam_angle,
                        z_axis=self.z_axis,
                        pixel_width=self.pixel_width_mm,
                        pixel_height=self.pixel_height_mm,
                        focal_length=self.focal_length_mm,
                        camera_coefficients=CameraCoefficients(
                            f=self.focal_length_mm,
                        )
                        )
                )
            except Exception as e:
                raise ValueError(f"Error creating ImageMeta for {image_id}: {e}")

            # Iterate through each row in the image group and create BoundingBox annotations
            cutout_id_counter = 0
            for _, row in group.iterrows():
                try:
                    class_id = self._class_id(row.get("classname"))
                    bbox_xywh = self._bbox_xywh(row, self.fullres_width, self.fullres_height)
                    annotation = BoundingBox(
                        bbox_xywh=bbox_xywh,
                        category_class_id=class_id,
                        cutout_id=f"{image_id}_{cutout_id_counter}",
                        local_coordinates=BBoxCoordinates(),
                        global_coordinates=GlobalCoordinates(
                            area_sqm=self._estimate_bbox_area_sqm(bbox_xywh),
                        ),
                    )
                    cutout_id_counter += 1
                    
                    image_meta.add_annotation(annotation)
                except Exception as e:
                    raise ValueError(f"Error processing row {row} for image {image_id}: {e}")

            # Save the metadata for this image  
            try:
                self._save_metadata(image_meta)
            except Exception as e:
                raise ValueError(f"Error saving metadata for {image_id}: {e}")
            
            log.info(f"Saved metadata for {image_id}")

    def _save_metadata(self, image_meta: ImageMeta) -> None:
        out_path = self.output_dir / f"{image_meta.image_id}.json"
        with open(out_path, "w") as f:
            json.dump(self._dataclass_to_dict(image_meta.to_dataclass()), f, indent=4)

    @staticmethod
    def _dataclass_to_dict(obj) -> Dict[str, Any]:
        """Recursively convert a dataclass to a dict for JSON serialization."""

        if isinstance(obj, list):
            return [RemapLabelsPipeline._dataclass_to_dict(i) for i in obj]
        elif hasattr(obj, "__dataclass_fields__"):
            return {k: RemapLabelsPipeline._dataclass_to_dict(v) for k, v in obj.__dict__.items()}
        else:
            return obj


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main entry point for the no remap labels pipeline."""
    try:
        pipeline = RemapLabelsPipeline(cfg)
        pipeline.process()
    except Exception as e:
        log.error(f"Error in RemapLabelsPipeline: {e}")
        raise
