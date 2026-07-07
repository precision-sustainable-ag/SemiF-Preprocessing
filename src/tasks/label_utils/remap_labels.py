import json
import logging
import random
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import hydra
from shapely.geometry import Polygon
from pyproj import Transformer
from tqdm import tqdm

import Metashape
from src.tasks.label_utils.filter_bboxes import BBoxFilter
from src.utils.utils import safe_save_json, extract_season_info, get_files, sanitize_time_for_path
from src.utils.datasets import (
    BBoxCoordinates,
    BoundingBox,
    CameraCoefficients,
    CameraInfo,
    FOV,
    GlobalCoordinates,
    ImageMetadata
)

log = logging.getLogger(__name__)


class DataMerger:
    def __init__(self, cfg: DictConfig) -> None:
        self.cam_angle = float(cfg.cam_angle)
        self.z_axis = float(cfg.z_axis)
        self.batch_dir = Path(cfg.paths.batch_dir)
        self.reference_dir = Path(cfg.paths.autosfm) / "reference"
        self.detections_dir = Path(cfg.paths.plant_detection_dir)
        self.output_path = Path(cfg.paths.autosfm) / "reference" / f"{self.batch_dir.name}_metadata.csv"

    def _load_csvs(self, directory: Path) -> pd.DataFrame:
        """
        Load all CSV files in a directory and concatenate them into a single DataFrame.
        Adds an 'image_id' column based on the file stem.
        If a file is empty, inserts a placeholder row with NaNs.

        Args:
            directory (Path): Directory containing CSVs to load.

        Returns:
            pd.DataFrame: Combined DataFrame of all CSVs.
        """
        csv_files = list(directory.glob("*.csv"))
        if not csv_files:
            log.error(f"No CSVs found in directory: {directory}")
            raise FileNotFoundError(f"No CSVs found in {directory}")
        
        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if df.empty:
                    empty_data = {col: None for col in df.columns}
                    empty_data["image_id"] = f.stem
                    empty_data["z_axis"] = self.z_axis
                    empty_data["cam_angle"] = self.cam_angle
                    empty_row = pd.DataFrame([empty_data])
                    dfs.append(empty_row)
                    log.debug(f"Empty CSV found and placeholder inserted: {f.name}")
                else:
                    df["image_id"] = f.stem
                    df["z_axis"] = self.z_axis
                    df["cam_angle"] = self.cam_angle
                    dfs.append(df)
            except Exception as e:
                log.error(f"Failed to read CSV: {f.name} - {e}")
                raise
        
        return pd.concat(dfs, ignore_index=True)

    def _load_reference_data(self) -> pd.DataFrame:
        """
        Load reference data for field of view and camera calibration.
        Merges on image label.
        """
        try:
            fov = pd.read_csv(self.reference_dir / "fov.csv")
            camera = pd.read_csv(self.reference_dir / "camera_reference.csv")
        except Exception as e:
            log.error(f"Failed to load reference CSVs: {e}")
            raise

        merged = camera.merge(fov, how="inner", on="label").rename(columns={"label": "image_id"})
        merged["is_normalized"] = True
        return merged

    def merge(self, save: bool = True) -> pd.DataFrame:
        """
        Merge detection data with camera reference data on image ID.
        """
        try:
            detections = self._load_csvs(self.detections_dir).rename(columns={"classname": "name"})
            detections["classifier_classname"] = None  # TODO: Replace with actual classifier
            detections["classifier_confidence"] = None  # TODO: Replace with actual confidence score

            reference = self._load_reference_data()
            merged = detections.merge(reference, on="image_id", how="left")

            if save:
                merged.to_csv(self.output_path, index=False)
                log.debug(f"Merged metadata saved to: {self.output_path}")

            return merged

        except Exception as e:
            log.error(f"Error during merge: {e}")
            raise

class BBoxMapper:
    def __init__(self, cfg: DictConfig, project_path: str, images: List[Dict]):
        """Class to map bounding box coordinates from image cordinates
        to global coordinates
        """
        self.batch_id = cfg.batch_id
        self.bbot_version = str(cfg.bbot_version)
        self.season = cfg.season
        self.crs = cfg.crs
        self.project_path = Path(project_path)
        self.images = images
        self.doc = Metashape.Document()
        self.doc.open(str(project_path), ignore_lock=True)
        self.camera_lookup: Dict[str, Metashape.Camera] = {
            cam.label: cam for chunk in self.doc.chunks for cam in chunk.cameras
        }
    
    def map(self) -> List[ImageMetadata]:
        """
        Maps bounding boxes for all images to global coordinates.

        Returns:
            List[ImageMetadata]: Updated image list with global coordinates for each bounding box.
        """
        # Create a list of chunks for each image_id
        log.info(f"Starting bounding box mapping for {len(self.images)} images using Metashape project: {self.project_path}")
        image_id_map = {img.image_id: [] for img in self.images}
        chunk = self._select_chunk(image_id_map)
        surface = chunk.model

        for img in tqdm(self.images, desc="Mapping 2D -> 3D"):
            updated = []
            log.debug(f"Processing image_id={img.image_id} with {len(img.annotations)} bounding boxes")

            for bbox in img.annotations:
                coords = self._map_bbox(
                    bbox, img.image_id, chunk, surface,
                    img.downscaled_height, img.downscaled_width
                )

                if coords is None:
                    log.warning(f"Mapping failed for image_id={img.image_id}, cutout_id={bbox.cutout_id}")
                    bbox.global_coordinates = self._default_global_coords()
                    continue

                bbox.global_coordinates = self._construct_global_coords(coords)

                updated.append(bbox)

            img.bboxes = updated
        log.info("Completed global coordinate mapping for all images.")
        return self.images

    def _select_chunk(self, image_map: Dict)  -> Metashape.Chunk:
        """Heuristically select the correct chunk from a project based on image presence."""
        labels = [chunk.label for chunk in self.doc.chunks]
        log.debug(f"Available chunks in project: {labels}")

        if "Merged Chunk" in labels:
            log.info("Using 'Merged Chunk' for mapping.")
            return self.doc.chunks[-1]
        
        if labels == ["Chunk 1"]:
            log.info("Using single 'Chunk 1'.")
            return self.doc.chunks[0]
        
        for chunk in self.doc.chunks:
            for cam in chunk.cameras:
                if cam.label in image_map:
                    image_map[cam.label].append(chunk)
        
        log.warning("Falling back to last chunk — heuristic may not be optimal.")
        return chunk
    
    def _map_bbox(self, bbox, image_id, chunk, surface, height, width):
        cam = self.camera_lookup.get(image_id)
        if not cam:
            log.warning(f"Camera not found for image_id={image_id}")
            return None

        mapped = []
        corners = ["top_left", "bottom_left", "top_right", "bottom_right"]
        coords = [getattr(bbox.local_coordinates,c) for c in corners]

        # Inward nudge steps to try if the first pickPoint misses the surface
        NUDGE_PX = [1, 3, 5, 10]

        for x, y in coords:
            px = float(np.clip(x * width,  1, width  - 1))
            py = float(np.clip(y * height, 1, height - 1))

            point = None
            # Try the original pixel, then nudge inward toward image center
            cx, cy = width / 2, height / 2
            dx = np.sign(cx - px)
            dy = np.sign(cy - py)

            for nudge in [0] + NUDGE_PX:
                nx = px + dx * nudge
                ny = py + dy * nudge
                ray_target = cam.unproject(Metashape.Vector([nx, ny]))
                
                if cam.center is None or ray_target is None:
                    log.warning(f"Camera center or ray target is None for image_id={image_id}, "
                                f"px={nx:.2f}, py={ny:.2f} - skipping this corner")
                    continue
                point = surface.pickPoint(cam.center, ray_target)
                if point is not None:
                    break

            if point is None:
                log.warning(
                    f"pickPoint failed after nudging for image_id={image_id}, "
                    f"x={px:.2f}, y={py:.2f} — skipping bbox"
                )
                return None  # Signal failure cleanly, no exception

            world_coord = chunk.transform.matrix.mulp(point)
            geo_coord = chunk.crs.project(world_coord)
            mapped.append([geo_coord.x, geo_coord.y])
        
        return [mapped[0], mapped[2], mapped[1], mapped[3]]
    
    def _calculate_area(self, coords: List[List[float]], from_latlon: bool = True) -> float:
        """
        Calculate polygon area in square meters from global coordinates.

        Args:
            coords (dict): Dict of bounding box corners (lat/lon).

        Returns:
            float: Area in square meters.
        """
        top_left =  coords[0]
        top_right =  coords[1]
        bottom_left =  coords[2]
        bottom_right =  coords[3]

        poly = Polygon([
            tuple(top_left),
            tuple(top_right),
            tuple(bottom_right),
            tuple(bottom_left),
            tuple(top_left)  # closing the polygon
        ])
        if from_latlon:
            gdf_proj = gpd.GeoDataFrame(index=[0], crs=self.crs, geometry=[poly])
            return gdf_proj.geometry[0].area
        else:
            return poly.area

    
    def _construct_global_coords(self, coords: List[List[float]]) -> GlobalCoordinates:
        """
        Constructs a GlobalCoordinates dataclass object from corner coordinates.
        Returns:
            GlobalCoordinates: Structured representation of the global bounding box.
        """
        centroid = [
            (coords[0][0] + coords[3][0]) / 2,
            (coords[0][1] + coords[3][1]) / 2
        ]
        if self.crs != "LOCAL":
            from_latlon = True
        else:
            from_latlon = False
        area = self._calculate_area(coords, from_latlon=from_latlon)

        return GlobalCoordinates(
            top_left=coords[0],
            top_right=coords[1],
            bottom_left=coords[2],
            bottom_right=coords[3],
            global_centroid=centroid,
            area_sqm=area
        )

    def _default_global_coords(self) -> GlobalCoordinates:
        return GlobalCoordinates(
            top_left=[0, 0],
            top_right=[0, 0],
            bottom_left=[0, 0],
            bottom_right=[0, 0],
            global_centroid=[0, 0],
            area_sqm=0.0
        )


class RemapLabels:
    def __init__(self, cfg: DictConfig):
        self.cfg = extract_season_info(cfg)
        self.batch_id = cfg.batch_id
        self.season = cfg.season
        self.bbot_version = cfg.bbot_version

        self.batch_dir = Path(cfg.paths.batch_dir)
        self.autosfm_dir = Path(cfg.paths.autosfm)
        self.merged_data_path = Path(cfg.paths.autosfm) / "reference" /  f"{self.batch_dir.name}_metadata.csv"
        self.metadata = pd.read_csv(self.merged_data_path)

        self.output_dir = self.batch_dir / "metadata"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.project_path = self.autosfm_dir / "project" / f"{cfg.batch_id}.psx"
        self.downscaled_dir = self.autosfm_dir / "downscaled_photos"
        self.fullres_dir = self.batch_dir / "images"
        self.fullres_h = cfg.exif.SVCamImageHeight if "3.1" in str(self.bbot_version) else cfg.exif.SonyImageHeight
        self.fullres_w = cfg.exif.SVCamImageWidth if "3.1" in str(self.bbot_version) else cfg.exif.SonyImageWidth

        self.sanitized_time = sanitize_time_for_path(cfg.start_time) if cfg.start_time else ""
        self.local_inspection_dir = (
            Path(cfg.paths.batch_dir) / "inspection" / self.sanitized_time
            if self.sanitized_time else Path(cfg.paths.batch_dir) / "inspection"
        )
        self.shp_dir = self.local_inspection_dir / "fov_shapefiles"
        self.shp_dir.mkdir(exist_ok=True, parents=True)

        with open(cfg.paths.species_info) as f:
            self.species_info = json.load(f)

        self.crs = cfg.crs

        log.info(f"Label projecting initialized for batch: {self.batch_id}, season: {self.season}, BBOT version: {self.bbot_version}")

    
    def _get_image_shape(self) -> Tuple[int, int]:
        """
        Returns the height and width of a sample image from the downscaled directory.
        """
        jpgs = list(self.downscaled_dir.glob("*.jpg")) + list(self.downscaled_dir.glob("*.JPG"))
        if not jpgs:
            log.error("No downscaled images found for shape estimation.")
            raise FileNotFoundError("No downscaled images found.")
        img_path = random.choice(jpgs)
        img = cv2.imread(str(img_path))
        log.debug(f"Using image {img_path.name} for shape reference.")
        return img.shape[:2]
    
    def _bbox_xywh(self, row: pd.Series) -> List[int]:
        """
        Convert normalized bounding box [xmin, ymin, xmax, ymax]
        into pixel coordinates [x, y, width, height].
        """
        x = round(row["xmin"] * self.fullres_w)
        y = round(row["ymin"] * self.fullres_h)
        w = round((row["xmax"] * self.fullres_w)) - x 
        h = round((row["ymax"] * self.fullres_h)) - y
        return [x, y, w, h]

    def _normalize_class_name(self, name: str) -> str:
        """Normalize detection class names for consistent lookup."""
        return str(name).strip().lower()

    def _build_metadata(self, image_id: str, h: int, w: int) -> ImageMetadata:
        """
        Build structured metadata for a single image including camera information
        and all associated bounding box annotations.

        Args:
            image_id (str): Unique ID of the image (typically the filename stem).
            h (int): Height of the downscaled image.
            w (int): Width of the downscaled image.

        Returns:
            ImageMetadata: Metadata object containing camera info and bounding boxes.
        """
        rows = self.metadata[self.metadata.image_id == image_id]
        if rows.empty:
            log.warning(f"No metadata rows found for image_id: {image_id}")
        
        try:
            camera_info = CameraInfo(
                aligned=rows["Alignment"].iloc[0],
                fov=self._fov(rows),
                z_axis=rows["z_axis"].iloc[0],
                cam_angle=rows["cam_angle"].iloc[0],
                estimated_xyz=self._camera_loc(rows),
                estimated_yaw=rows["Estimated_Yaw"].iloc[0],
                estimated_pitch=rows["Estimated_Pitch"].iloc[0],
                estimated_roll=rows["Estimated_Roll"].iloc[0],
                pixel_width=rows["pixel_width"].iloc[0],
                pixel_height=rows["pixel_height"].iloc[0],
                focal_length=rows["f"].iloc[0],
                camera_coefficients=CameraCoefficients(
                    f=rows["f"].iloc[0],
                    cx=rows["cx"].iloc[0],
                    cy=rows["cy"].iloc[0],
                    b1=rows["b1"].iloc[0],
                    b2=rows["b2"].iloc[0],
                    k1=rows["k1"].iloc[0],
                    k2=rows["k2"].iloc[0],
                    k3=rows["k3"].iloc[0],
                    k4=rows["k4"].iloc[0],
                    p1=rows["p1"].iloc[0],
                    p2=rows["p2"].iloc[0]
                )
            )
        except Exception as e:
            log.exception(f"Failed to build CameraInfo for {image_id}: {e}")
            raise 

        bboxes = []
        for _, row in rows.iterrows():
            # if the xmin, ymin, etc are not None
            if pd.isna(row["xmin"]) or pd.isna(row["ymin"]) or pd.isna(row["xmax"]) or pd.isna(row["ymax"]):
                log.debug(f"Skipping invalid bbox for {image_id}: missing coords.")
                continue
            try:
                local_coords = BBoxCoordinates(
                    top_left=[row["xmin"], row["ymin"]],
                    top_right=[row["xmax"], row["ymin"]],
                    bottom_left=[row["xmin"], row["ymax"]],
                    bottom_right=[row["xmax"], row["ymax"]],
                    local_centroid=[(row["xmin"] + row["xmax"]) / 2, (row["ymin"] + row["ymax"]) / 2],
                    is_normalized=row["is_normalized"],
                )
                
                detection_name = self._normalize_class_name(row.get("name", ""))
                species_entry = next(
                    (v for k, v in self.species_info["species"].items()
                    if self._normalize_class_name(k) == detection_name
                    or detection_name in [self._normalize_class_name(a) for a in v.get("alias", [])]),
                    {}
                )
                category_class_id = species_entry.get("class_id", None)
                if category_class_id is None:
                    log.warning(f"Unknown detection class '{row.get('name')}' for {image_id}")

                bbox = BoundingBox(
                    is_primary=None,
                    cutout_exists=None,
                    bbox_xywh=self._bbox_xywh(row),
                    # image_id=image_id,
                    category_class_id=category_class_id,
                    detection_class=detection_name, 
                    cutout_id=f"{image_id}_{row['bounding_box_id']}",
                    # overlapping_cutout_ids=None,
                    local_coordinates=local_coords,
                    non_target_weed=row["classifier_classname"],
                    non_target_weed_pred_conf=row["classifier_confidence"]
                )
                bboxes.append(bbox)
            except Exception as e:
                log.exception(f"Failed to create bounding box for {image_id}: {e}")

        log.debug(f"Built metadata for {image_id}: {len(bboxes)} bounding boxes")

        return ImageMetadata(
            season=self.season,
            datetime=None,
            bbot_version=str(self.bbot_version),
            image_id=image_id,
            batch_id=self.batch_id,
            validated=False,
            version="1",
            exif_meta=None,
            camera_info=camera_info,
            annotations=bboxes,
            fullres_width=self.fullres_w,
            downscaled_height=h,
            downscaled_width=w,
            fullres_height=self.fullres_h,
        )
    
    def _check_fov_coords(self, coords: List[Tuple[float, float]] | None) -> bool:
        """
        Check if the coordinates are valid for FOV.
        Returns True if valid, False otherwise.
        """
        if not coords or coords is None:
            return False
        # If any None in the list or any element is not a tuple of two floats
        for pt in coords:
            if (
                pt is None or
                len(pt) != 2 or
                pt[0] is None or pt[1] is None or
                (isinstance(pt[0], float) and np.isnan(pt[0])) or
                (isinstance(pt[1], float) and np.isnan(pt[1]))
            ):
                return False
        return True
    
    def calculate_area(self, coords: list[Tuple[float, float]] | None) -> float:
        """
        Calculate polygon area in square meters from WGS84 coordinates.
        Returns 0.0 if coords is None or contains None/invalid values.
        """        
        # If any None in the list or any element is not a tuple of two floats
        if not self._check_fov_coords(coords):
            return None
            
        if len(coords) < 3:
            raise ValueError("At least 3 coordinates are required to form a polygon.")

        # transformer = Transformer.from_crs(self.crs, self.crs, always_xy=True)
        # coords_t = [transformer.transform(*pt) for pt in coords]
        coords_t = coords

        # Ensure polygon is closed
        if coords_t[0] != coords_t[-1]:
            coords_t.append(coords_t[0])

        try:
            poly = Polygon(coords_t) 
        except Exception as e:
            log.exception(f"Failed to create polygon from coordinates: {e}")
            raise

        return poly.area


    def calculate_fov_area(self, fov: Dict) -> float:
        try:
            corners = [
                fov["top_left_xy"],
                fov["top_right_xy"],
                fov["bottom_right_xy"],
                fov["bottom_left_xy"],
            ]
            if "3" in str(self.bbot_version):
                # For non-TX batches, we need to calculate the area in meters
                area = self.calculate_area(corners) 
                area_cm2 = area * 10000 if area else None
            else:
                if not self._check_fov_coords(corners):
                    return None
                polygon = Polygon(corners)
                area_cm2 = polygon.area * 10000
            return area_cm2
        
        except Exception as e:
            log.exception(f"Failed to calculate FOV area: {e}")
            raise
        
    def _fov(self, rows: pd.DataFrame) -> FOV:
        """
        Extract the camera field-of-view (FOV) parameters from a row group.
        """
        try:
            fov = {
            "top_left_xy":(rows["top_left_x"].iloc[0], rows["top_left_y"].iloc[0]),
            "top_right_xy":(rows["top_right_x"].iloc[0], rows["top_right_y"].iloc[0]),
            "bottom_left_xy":(rows["bottom_left_x"].iloc[0], rows["bottom_left_y"].iloc[0]),
            "bottom_right_xy":(rows["bottom_right_x"].iloc[0], rows["bottom_right_y"].iloc[0])
            }
            return FOV(
                height=rows["height"].iloc[0],
                width=rows["width"].iloc[0],
                top_left_xy=[rows["top_left_x"].iloc[0], rows["top_left_y"].iloc[0]],
                top_right_xy=[rows["top_right_x"].iloc[0], rows["top_right_y"].iloc[0]],
                bottom_left_xy=[rows["bottom_left_x"].iloc[0], rows["bottom_left_y"].iloc[0]],
                bottom_right_xy=[rows["bottom_right_x"].iloc[0], rows["bottom_right_y"].iloc[0]],
                fov_area_cm2=self.calculate_fov_area(fov)  # To be calculated or updated elsewhere
            ) 
        except Exception as e:
            log.exception(f"Failed to construct FOV object for {rows['image_id'].iloc[0]}.")
            raise
    
    def _camera_loc(self, rows: pd.DataFrame) -> List[float]:
        """
        Extract estimated 3D camera location from metadata.
        """
        try:
            return [
                rows["Estimated_X"].iloc[0],
                rows["Estimated_Y"].iloc[0],
                rows["Estimated_Z"].iloc[0]
            ]
        except Exception as e:
            log.exception("Failed to extract camera location.")
            raise
    
    def remap(self) -> List[Dict]:
        """
        Main remapping procedure that:
        1. Builds metadata for all images
        2. Maps bounding boxes from image coordinates to global coordinates via BBoxMapper
        3. Sorts the output by image ID

        Returns:
            List[dict]: List of updated ImageMetadata objects with mapped global coordinates
        """
        log.info("Starting metadata remapping process...")

        try:
            h, w = self._get_image_shape()
            log.debug(f"Detected downscaled image shape: height={h}, width={w}")
            unique_image_ids = get_files(self.cfg, task="remap_labels")
            log.info(f"Found {len(unique_image_ids)} unique image IDs.")
            images = [self._build_metadata(iid, h, w) for iid in unique_image_ids]
            log.info("Successfully built ImageMetadata objects.")
        except Exception as e:
            log.exception("Failed to build ImageMetadata objects.")
            raise

        # Sort to ensure consistent order
        images.sort(key=lambda x: x.image_id)

        try:
            mapper = BBoxMapper(self.cfg, self.project_path, images)
            mapped_images = mapper.map()
            log.info("Successfully mapped bounding boxes to global coordinates.")
            return mapped_images
        except Exception as e:
            log.exception("Mapping bounding boxes to global coordinates failed.")
            raise

    # Optional: convert numpy types if needed
    def sanitize(self, obj):
        if isinstance(obj, dict):
            return {k: self.sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.sanitize(i) for i in obj]
        elif isinstance(obj, (np.bool_, np.int_, np.float64)):
            return obj.item()  # convert to native Python type
        return obj
    
    def save_image_json(self, image: ImageMetadata) -> None:
        """Save a single image's metadata as a JSON file to the metadata directory."""
        path = self.output_dir / f"{image.image_id}.json"
        try:
            # Convert to dictionary and sanitize for JSON compatibility
            image_dict = self.sanitize(asdict(image))
            safe_save_json(image_dict, path)

            log.debug(f"Image {image.image_id}: saved {len(image.annotations)} annotations.")
        except Exception as e:
            log.exception(f"Failed to save image metadata for {image.image_id} to {path}")
            raise

    def save_bbox_fov_shp(self, images_data: List[ImageMetadata]) -> None:
        """
        Save shapefiles for both image field-of-view (FOV) and bounding boxes.

        Args:
            images_data (List[ImageMetadata]): List of images with mapped metadata.
        """
        log.info("Starting shapefile export for FOVs and bounding boxes.")
        fov_records = []

        for image_data in images_data:
            image_id = image_data.image_id
            fov_coords = image_data.camera_info.fov
            
            if not fov_coords:
                log.warning(f"Missing FOV for image {image_id}. Skipping FOV.")
                continue
            
            try:
                fov_polygon = self._build_polygons(fov_coords, fov=True)
                fov_records.append({
                    "image_id": image_id,
                    "camera_x": image_data.camera_info.estimated_xyz[0],
                    "camera_y": image_data.camera_info.estimated_xyz[1],
                    "camera_z": image_data.camera_info.estimated_xyz[2],
                    "yaw": image_data.camera_info.estimated_yaw,
                    "pitch": image_data.camera_info.estimated_pitch,
                    "roll": image_data.camera_info.estimated_roll,
                    "geometry": fov_polygon,
                })
            except Exception as e:
                log.warning(f"[{image_id}] Failed to build FOV polygon: {e}")
                continue

        fov_output_path = self.shp_dir / f"{self.batch_dir.name}_image_fovs.shp"
        self._write_shapefile(fov_records, fov_output_path)

        log.info("Completed shapefile export.")

    def _build_polygons(self, coords: object, fov: bool = False) -> Polygon:
        """
        Build a shapely Polygon from bounding box or FOV coordinates.
        """
        
        if fov:
            suffix = "_xy"
        else:
            suffix = ""
            
        return Polygon([
            tuple(getattr(coords,f"top_left{suffix}")),
            tuple(getattr(coords,f"top_right{suffix}")),
            tuple(getattr(coords,f"bottom_right{suffix}")),
            tuple(getattr(coords,f"bottom_left{suffix}")),
            tuple(getattr(coords,f"top_left{suffix}"))  # close the polygon
        ])
        
    
    def _write_shapefile(self, records: List[Dict], output_path: Path) -> None:
        """
        Write a list of record dictionaries to an ESRI Shapefile.

        Args:
            records (List[dict]): Records with geometry and attributes.
            output_path (Path): Where the shapefile should be saved.
        """
        try:
            gdf = gpd.GeoDataFrame(records, crs=self.crs)
            gdf.to_file(output_path, driver='ESRI Shapefile')
            log.info(f"Saved shapefile: {output_path} with {len(records)} features.")
        except Exception as e:
            log.exception(f"Failed to write shapefile: {output_path}")
            raise
        

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Full pipeline for remapping plant bounding boxes:
      1. Merge detection and camera metadata
      2. Construct image metadata and map bboxes to global coordinates
      3. Filter and deduplicate bounding boxes
      4. Save metadata as shapefiles and per-image JSON files

    Args:
        cfg (DictConfig): Configuration object from Hydra.
    """
    start = time.time()
    log.info("Starting RemapLabels pipeline...")

    try:
        merger = DataMerger(cfg)
        merged_df = merger.merge()
        log.info(f"Merged metadata: {merged_df.shape[0]} rows.")
    except Exception as e:
        log.exception("Metadata merging failed.")
        raise

    try:
        remapper = RemapLabels(cfg)
        images = remapper.remap()
    except Exception as e:
        log.exception("Remapping metadata failed.")
        raise

    try:
        bbox_filter = BBoxFilter(cfg, images)
        bbox_filter.deduplicate_bboxes()
        log.info("Bounding box deduplication completed.")
    except Exception as e:
        log.exception("Bounding box filtering failed.")
        raise

    try:
        imgs = bbox_filter.images
        remapper.save_bbox_fov_shp(imgs)
        for img in imgs:
            remapper.save_image_json(img)
        log.info(f"Successfully saved metadata for {len(imgs)} images.")
    except Exception as e:
        log.exception("Saving output files failed.")
        raise

    end = time.time()
    log.info(f"RemapLabels pipeline completed in {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()