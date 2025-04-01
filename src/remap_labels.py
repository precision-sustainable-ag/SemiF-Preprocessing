import json
import logging
import random
import time
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import cv2
import geopandas as gpd
import pandas as pd
from omegaconf import DictConfig
from shapely.geometry import Polygon
from pyproj import CRS
from tqdm import tqdm
from utils.datasets import BBoxCoordinates, GlobalCoordinates, BoundingBox, CameraInfo, ImageMetadata, FOV, CameraCoefficients
from dataclasses import asdict
from datetime import datetime
import Metashape
from filter_bboxes import BBoxFilter

log = logging.getLogger(__name__)


class DataMerger:
    def __init__(self, cfg: DictConfig) -> None:
        self.batch_dir = Path(cfg.paths.batch_dir)
        self.reference_dir = Path(cfg.paths.autosfm) / "reference"
        self.detections_dir = self.batch_dir / "plant-detections"
        self.output_path = Path(cfg.paths.autosfm) / "reference" / f"{self.batch_dir.name}_metadata.csv"

    def _load_csvs(self, directory: Path) -> pd.DataFrame:
        csv_files = list(directory.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSVs found in {directory}")
        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            if df.empty:
                empty_data = {col: None for col in df.columns}
                empty_data["image_id"] = f.stem
                empty_row = pd.DataFrame([empty_data])
                dfs.append(empty_row)
            else:
                df["image_id"] = f.stem
                dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)

    def _load_reference_data(self) -> pd.DataFrame:
        fov = pd.read_csv(self.reference_dir / "fov.csv")
        camera = pd.read_csv(self.reference_dir / "camera_reference.csv")
        merged = camera.merge(fov, how="inner", on="label").rename(columns={"label": "image_id"})
        merged["is_normalized"] = True
        return merged

    def merge(self, save: bool = True) -> pd.DataFrame:
        detections = self._load_csvs(self.detections_dir).rename(columns={"classname": "name"})
        detections["classifier_classname"] = "dummy_classifier_name" # TODO: fix this
        detections["classifier_confidence"] = 0.999
        reference = self._load_reference_data()
        merged = detections.merge(reference, on="image_id", how="left")
        if save:
            merged.to_csv(self.output_path, index=False)
        return merged

class BBoxMapper:
    def __init__(self, project_path: str, images: List[dict]):
        """Class to map bounding box coordinates from image cordinates
        to global coordinates
        """
        self.project_path = project_path
        self.images = images
        self.doc = Metashape.Document()
        self.doc.open(str(project_path), ignore_lock=True)
    
    def map(self):
        """
        Maps all the bounding boxes to a global coordinate space
        """
        # Create a list of chunks for each image_id
        image_id_map = {img.image_id: [] for img in self.images}
        chunk = self._select_chunk(image_id_map)
        surface = chunk.model

        for img in tqdm(self.images, desc="Mapping 2D -> 3D"):
            updated = []
            for bbox in img.annotations:
                try:
                    coords = self._map_bbox(bbox, img.image_id, chunk, surface, img.downscaled_height, img.downscaled_width)
                    bbox.global_coordinates = self._construct_global_coords(coords)

                except Exception as e:
                    bbox.global_coordinates = self._default_global_coords()
                    log.exception(f"Mapping failed for {img.image_id}: {e}")
                updated.append(bbox)

            img.bboxes = updated
        return self.images

    def _select_chunk(self, image_map: dict):
        labels = [chunk.label for chunk in self.doc.chunks]
        if "Merged Chunk" in labels:
            return self.doc.chunks[-1]
        if labels == ["Chunk 1"]:
            return self.doc.chunks[0]
        for chunk in self.doc.chunks:
            for cam in chunk.cameras:
                if cam.label in image_map:
                    image_map[cam.label].append(chunk)
        return chunk
    
    def _map_bbox(self, bbox, image_id, chunk, surface, height, width):
        cam = next((c for c in chunk.cameras if c.label == image_id), None)
        
        if not cam:
            raise ValueError(f"No camera found for {image_id}")

        mapped = []
        corners = ["top_left", "bottom_left", "top_right", "bottom_right"]
        coords = [getattr(bbox.local_coordinates,c) for c in corners]
        
        for x, y in coords:
            px = max(x * width, 0)
            py = max(y * height, 0)
            ray_target = cam.unproject(Metashape.Vector([px, py]))
            point = surface.pickPoint(cam.center, ray_target)

            if point is None:
                raise ValueError(f"pickPoint failed for {image_id}")

            world_coord = chunk.transform.matrix.mulp(point)
            geo_coord = chunk.crs.project(world_coord)
            mapped.append([geo_coord.x, geo_coord.y])
        
        return [mapped[0], mapped[2], mapped[1], mapped[3]]

    def _calculate_area_from_latlon(self, coords: dict) -> float:
        """
        Calculates area in square meters from lat/lon global coordinates.
        """
        # Create a polygon
        poly = Polygon([
            tuple(coords["top_left"]),
            tuple(coords["top_right"]),
            tuple(coords["bottom_right"]),
            tuple(coords["bottom_left"]),
            tuple(coords["top_left"])  # close the polygon
        ])

        # Put polygon into GeoDataFrame with WGS84 CRS
        gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly])

        # Step 3: Reproject to UTM zone appropriate for the location
        # EPSG:32617 is UTM zone 17N (covers -78 longitude; check for other zones as needed)
        gdf_proj = gdf.to_crs(CRS("EPSG:32617"))

        # Compute area
        area_sqm = gdf_proj.geometry[0].area
        return area_sqm
    def _construct_global_coords(self, coords: List[List[float]]) -> dict:
        centroid = [(coords[0][0] + coords[3][0]) / 2, (coords[0][1] + coords[3][1]) / 2]
        return GlobalCoordinates(
            top_left=coords[0], 
            top_right=coords[1],
            bottom_left=coords[2], 
            bottom_right=coords[3],
            global_centroid=centroid,
            area_sqm=self._calculate_area_from_latlon({
                "top_left": coords[0], "top_right": coords[1],
                "bottom_left": coords[2], "bottom_right": coords[3]
                }))

    def _default_global_coords(self) -> dict:
        return {
            "top_left": [0, 0], "top_right": [0, 0],
            "bottom_left": [0, 0], "bottom_right": [0, 0],
            "global_centroid": [0, 0], "area_sqm": 0.0
        }


class RemapLabels:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.batch_id = cfg.batch_id
        self.season_cfg = cfg.date_ranges
        self.season, self.bbot_version = self._get_season_and_bbot()
        self.batch_dir = Path(cfg.paths.batch_dir)
        self.autosfm_dir = Path(cfg.paths.autosfm)
        self.merged_data_path = Path(cfg.paths.autosfm) / "reference" /  f"{self.batch_dir.name}_metadata.csv"
        self.metadata = pd.read_csv(self.merged_data_path)
        self.output_dir = self.batch_dir / "metadata"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.project_path = self.autosfm_dir / "project" / f"{cfg.batch_id}.psx"
        self.downscaled_dir = self.autosfm_dir / "downscaled_photos"
        self.fullres_dir = self.batch_dir / "images"
        self.fullres_h = cfg.exif.Image.ImageHeight
        self.fullres_w = cfg.exif.Image.ImageWidth

        self.shp_dir = cfg.paths.fov_shapefiles
        self.shp_dir.mkdir(exist_ok=True, parents=True)

    def _get_image_shape(self) -> tuple:
        jpgs = list(self.downscaled_dir.glob("*.jpg")) + list(self.downscaled_dir.glob("*.JPG"))
        img = cv2.imread(str(random.choice(jpgs)))
        return img.shape[:2]
    
    def _bbox_xywh(self, row: pd.Series) -> List[int]:
        x = round(row["xmin"] * self.fullres_w)
        y = round(row["ymin"] * self.fullres_h)
        w = round((row["xmax"] * self.fullres_w)) - x 
        h = round((row["ymax"] * self.fullres_h)) - y
        return [x, y, w, h]

    def _get_season_and_bbot(self) -> Optional[str]:

        try:
            site, date_str = self.batch_id.split("_")
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
            site_seasons = getattr(self.season_cfg, site)

            for season_name, season_range in site_seasons.items():
                start = datetime.strptime(season_range["start"], "%Y-%m-%d").date()
                end = datetime.strptime(season_range["end"], "%Y-%m-%d").date()
                bbot_version = season_range.get("bbot_version", None)
                if start <= date <= end:
                    return season_name, bbot_version

            return None, None  # No matching season
        except Exception as e:
            log.exception(f"Error parsing batch ID {self.batch_id}: {e}")
            exit()
            return None, None
        

    def _build_metadata(self, image_id: str, h: int, w: int) -> ImageMetadata:
        rows = self.metadata[self.metadata.image_id == image_id]
        camera_info = CameraInfo(
            aligned=rows["Alignment"].iloc[0],
            fov=self._fov(rows),
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

        bboxes = []
        for _, row in rows.iterrows():
            # if the xmin, ymin, etc are not None
            if pd.isna(row["xmin"]) or pd.isna(row["ymin"]) or pd.isna(row["xmax"]) or pd.isna(row["ymax"]):
                continue
            local_coords = BBoxCoordinates(
                top_left=[row["xmin"], row["ymin"]],
                top_right=[row["xmax"], row["ymin"]],
                bottom_left=[row["xmin"], row["ymax"]],
                bottom_right=[row["xmax"], row["ymax"]],
                local_centroid=[(row["xmin"] + row["xmax"]) / 2, (row["ymin"] + row["ymax"]) / 2],
                is_normalized=row["is_normalized"],
            )
            bbox = BoundingBox(
                is_primary=None,
                cutout_exists=None,
                bbox_xywh=self._bbox_xywh(row),
                # image_id=image_id,
                category_class_id=None,
                cutout_id=f"{image_id}_{row['bounding_box_id']}",
                # overlapping_cutout_ids=None,
                local_coordinates=local_coords,
                non_target_weed=row["classifier_classname"],
                non_target_weed_pred_conf=row["classifier_confidence"]
            )
            bboxes.append(bbox)

        return ImageMetadata(
            season=self.season,
            datetime="",
            bbot_version=self.bbot_version,
            image_id=image_id,
            batch_id=self.batch_id,
            validated=False,
            version="v1",
            Exif_meta="",
            camera_info=camera_info,
            annotations=bboxes,
            
            fullres_width=self.fullres_w,
            downscaled_height=h,
            downscaled_width=w,
            fullres_height=self.fullres_h,
        )
    
    def _bbox_coords(self, row) -> dict:
        tl = [row["xmin"], row["ymin"]]
        br = [row["xmax"], row["ymax"]]
        centroid = [(tl[0] + br[0]) / 2, (tl[1] + br[1]) / 2]
        return {
            "top_left": tl, "top_right": [br[0], tl[1]],
            "bottom_left": [tl[0], br[1]], "bottom_right": br,
            "local_centroid": centroid,
            "is_normalized": row["is_normalized"]
        }

    def _fov(self, rows):
        return FOV(
            height=rows["height"].iloc[0],
            width=rows["width"].iloc[0],
            top_left_xy=[rows["top_left_x"].iloc[0], rows["top_left_y"].iloc[0]],
            top_right_xy=[rows["top_right_x"].iloc[0], rows["top_right_y"].iloc[0]],
            bottom_left_xy=[rows["bottom_left_x"].iloc[0], rows["bottom_left_y"].iloc[0]],
            bottom_right_xy=[rows["bottom_right_x"].iloc[0], rows["bottom_right_y"].iloc[0]],
            fov_area_m2=None
        )
    
    def _camera_loc(self, rows):
        return [rows["Estimated_X"].iloc[0], rows["Estimated_Y"].iloc[0], rows["Estimated_Z"].iloc[0]]
    
    def remap(self) -> List[dict]:
        h, w = self._get_image_shape()
        images = [self._build_metadata(iid, h, w) for iid in sorted(self.metadata["image_id"].unique())]
        return BBoxMapper(self.project_path, images).map()

    # Optional: convert numpy types if needed
    def sanitize(self, obj):
        if isinstance(obj, dict):
            return {k: self.sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.sanitize(i) for i in obj]
        elif isinstance(obj, (np.bool_, np.int_, np.float64)):
            return obj.item()  # convert to native Python type
        return obj
    
    def save_image_json(self, image: ImageMetadata):
        path = self.output_dir / f"{image.image_id}.json"
        with open(path, "w") as f:
            image = asdict(image)
            # for box in image["annotations"]:
                # remove global coordinates attribute completely from the bounding box
                # box.pop("global_coordinates")
                # box.pop("local_coordinates")
            json.dump(self.sanitize(image), f, indent=4)
        log.info(f"Saved: {path}")

    def _build_polygons(self, coords: dict, fov: bool= False) -> Polygon:
        
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
        
    
    def _write_shapefile(self, records: List[dict], output_path: Path):
        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
        gdf.to_file(output_path, driver='ESRI Shapefile')
        log.info(f"Saved shapefile: {output_path}")
        

    def save_bbox_fov_shp(self, images_data: List[dict]):
        """
        Save all bounding boxes from all images into one shapefile.
        Each bbox becomes one polygon with metadata.
        """
        bbox_records = []
        fov_records = []

        for image_data in images_data:
            image_id = image_data.image_id
            fov_coords = image_data.camera_info.fov
            
            if not fov_coords:
                log.warning(f"Missing FOV for image {image_id}. Skipping FOV.")
                continue
            
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

            for bbox in image_data.annotations:
                coords = bbox.global_coordinates
                
                if not coords:
                    log.warning(f"Missing global coordinates for bbox {bbox.cutout_id} in image {image_id}. Skipping.")
                    continue  # skip unmapped bbox

                bbox_polygon = self._build_polygons(coords)

                bbox_records.append({
                    "image_id": image_id,
                    "cutout_id": bbox.cutout_id,
                    "geometry": bbox_polygon,
                    "area_sqm": coords.area_sqm,
                    "centroid": coords.global_centroid,
                })

        bbox_output_path = self.shp_dir / f"{self.batch_dir.name}_bboxes_fov.shp"
        fov_output_path = self.shp_dir / f"{self.batch_dir.name}_image_fovs.shp"
        self._write_shapefile(bbox_records, bbox_output_path)
        self._write_shapefile(fov_records, fov_output_path)

        
def main(cfg: DictConfig) -> None:
    # TODO: documentation
    start = time.time()
    
    merger = DataMerger(cfg)
    df = merger.merge()
    
    remapper = RemapLabels(cfg)
    images = remapper.remap()
    
    bbox_filter = BBoxFilter(cfg, images)
    bbox_filter.deduplicate_bboxes()
    imgs = bbox_filter.images
    
    remapper.save_bbox_fov_shp(imgs)
    
    for img in imgs:
        remapper.save_image_json(img)

    end = time.time()
    log.info(f"Remap completed in {end - start} seconds.")

if __name__ == "__main__":
    main()