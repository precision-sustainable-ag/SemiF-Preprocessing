import json
import logging
from pathlib import Path
from typing import List, Dict
import geopandas as gpd
import numpy as np
from omegaconf import DictConfig
import hydra
from shapely.geometry import Point, Polygon
from tqdm import tqdm

from src.utils.utils import safe_save_json, get_files

log = logging.getLogger(__name__)

class SpeciesAssigner :
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.batch_id = cfg.batch_id
        self.season = cfg.season
        self.spec_dict = self.read_json(cfg.paths.species_info)
        self.metadata_path = Path(cfg.paths.batch_dir, "metadata")
        self.output_shp_dir = Path(cfg.paths.inspection_dir, "fov_shapefiles")
        self.output_shp_dir.mkdir(parents=True, exist_ok=True)
        self.output_shp_path = self.output_shp_dir / f"{self.batch_id}_bboxes_fov.shp"
        
        self.shapefile_path = Path(cfg.paths.semif_util_dir) / "autosfm" / "ShapeFiles" / self.season / f"{self.season}.shp"
        self.polygons = gpd.read_file(self.shapefile_path).to_crs(cfg.crs)
        self.closest_distance_thresh = 2  # meters

        self.bbox_polygons = []

        log.info(f"Initialized SpeciesAssigner for batch: {self.batch_id}, season: {self.season}")
        log.info(f"Loaded shapefile from: {self.shapefile_path}")

    def read_json(self, filepath: Path) -> Dict:
        with open(filepath) as f:
            metadata = json.load(f)
        return metadata
    
    def save_json(self, filepath: Path, data: Dict) -> None:
        safe_save_json(data, filepath)
    
    def run(self) -> None:
        """
        Assign species labels to all bounding boxes found in image metadata.
        """
        metadata_files = get_files(self.cfg, task="assign_species")
        log.info(f"Found {len(metadata_files)} metadata files to process.")
        for file in tqdm(metadata_files, desc="Assigning labels"):
            self._process_file(file)
        self.save_bbox_shapefile()


    def save_bbox_shapefile(self) -> None:
        if not self.bbox_polygons:
            log.warning("No bbox polygons collected. Skipping shapefile save.")
            return
        gdf = gpd.GeoDataFrame(self.bbox_polygons, crs=self.cfg.crs)
        gdf.to_file(self.output_shp_path)
        log.info(f"Saved bbox polygons shapefile: {self.output_shp_path}")

    def _process_file(self, filepath: Path)  -> None:
        """Load and process a single image metadata file."""
        metadata = self.read_json(filepath)
        image_id = metadata.get("image_id", "")
        batch_id = metadata.get("batch_id", "")
        
        for bbox in metadata.get("annotations", []):
            species_info = self._determine_species(bbox, batch_id)
            self._assign_species(bbox, species_info)

            # Collect bbox polygon info for shapefile
            if "global_coordinates" in bbox:
                poly = self._bbox_to_polygon(bbox["global_coordinates"])
                comm_name = species_info.get("common_name", "")
                cutout_id = bbox.get("cutout_id", "")
                area_sqm = bbox.get("area_sqm", 0)
                global_centroid = bbox.get("global_coordinates", {}).get("global_centroid", [0, 0])
                is_primary = bbox.get("is_primary", None)
                category_class_id = bbox.get("category_class_id", None)
                non_target_weed = bbox.get("non_target_weed", None)
                non_target_weed_pred_conf = bbox.get("non_target_weed_pred_conf", None)
                self.bbox_polygons.append({
                    "batch_id": batch_id,
                    "image_id": image_id,
                    "cutout_id": cutout_id,
                    "comm_name": comm_name,
                    "category_c": category_class_id,
                    "is_primary": is_primary,
                    "non_target": non_target_weed,
                    "non_targ_1": non_target_weed_pred_conf,
                    "area_sqm": area_sqm,
                    "geometry": poly,
                    "centroid": global_centroid,
                })

        self.save_json(filepath, metadata)
        log.debug(f"Updated species labels in: {filepath.name}")

    def _bbox_to_polygon(self, global_coordinates: Dict) -> Polygon:
        # Expects a dict with keys: top_left, top_right, bottom_right, bottom_left (order matters)
        corners = [
            tuple(global_coordinates["top_left"]),
            tuple(global_coordinates["top_right"]),
            tuple(global_coordinates["bottom_right"]),
            tuple(global_coordinates["bottom_left"]),
            tuple(global_coordinates["top_left"]),  # Close polygon
        ]
        return Polygon(corners)
    
    def _determine_species(self, bbox: Dict, batch_id: str) -> Dict:
        """
        Determine species based on spatial location or fallback rules.
        """
        x, y = bbox["global_coordinates"]["global_centroid"]
        bbox_cls = bbox.get("category_class_id")
        if bbox_cls == 28:
            return self.spec_dict["species"].get("colorchecker", None)

        if "cash" in self.season and bbox_cls != 28:
            return self._get_cash_crop_species()

        point = Point(x, y)
        return self._lookup_species_from_point(point, bbox, batch_id)

    def _get_cash_crop_species(self) -> Dict:
        """Assign hardcoded species for known cash crop batches."""
        batch_prefix = self.batch_id
        crop_lookup = {"NC": "GLMA4", "MD": "ZEA", "TX": "GOHI"}
        for prefix, species_code in crop_lookup.items():
            if prefix in batch_prefix:
                return self.spec_dict["species"].get(species_code, self.spec_dict["species"]["plant"])
        return self.spec_dict["species"]["plant"]

    def _lookup_species_from_point(self, point: Point, bbox: Dict, batch_id: str) -> Dict:
        """Match bounding box centroid to polygon to infer species."""
        contains_point = self.polygons["geometry"].apply(lambda poly: poly.contains(point))
        containing = self.polygons[contains_point]

        if not len(containing):
            return self._handle_no_polygon_match(point, bbox)
        
        poly_cls = containing["species"].values[0]
        comm_name = containing["comm_name"].values[0]
        class_id = containing["class_id"].values[0]

        if all(val in [None, np.nan] for val in [comm_name, class_id]):
            return self._handle_undefined_species(containing, bbox, batch_id)

        return self.spec_dict["species"].get(poly_cls, self.spec_dict["species"]["plant"])

    def _handle_no_polygon_match(self, point: Point, bbox: Dict) -> Dict:
        bbox_id = bbox.get("cutout_id", "unknown")
        log.warning(f"No polygon found for bbox_id: {bbox_id}. Trying closest.")
        distances = self.polygons["geometry"].apply(lambda poly: poly.distance(point))
        closest_idx = distances.idxmin()
        closest_distance = distances[closest_idx]

        if closest_distance < self.closest_distance_thresh:
            closest_poly = self.polygons.loc[closest_idx]
            log.warning(f"Using closest polygon '{closest_poly['comm_name']}' for bbox '{bbox_id}'")
            return self.spec_dict["species"].get(closest_poly["species"], self.spec_dict["species"]["plant"])

        log.warning(f"No nearby polygon within {self.closest_distance_thresh}m for bbox '{bbox_id}'. Using fallback species.")
        return self.spec_dict["species"]["plant"]

    def _handle_undefined_species(self, containing: List[Polygon], bbox: Dict) -> Dict:
        poly_id = containing["id"].values[0]
        bbox_id = bbox.get("cutout_id", "unknown")
        log.warning(f"Polygon {poly_id} has no defined species. Assigning bbox {bbox_id} as non_target_weed.")


        if bbox.get("non_target_weed") != "non_target_weed":
            bbox["non_target_weed"] = "non_target_weed"
            bbox["non_target_weed_pred_conf"] = 1.0

        return self.spec_dict["species"]["plant"]

    def _assign_species(self, bbox: Dict, species_info: Dict):
        bbox.update({
            "category_class_id": species_info.get("class_id"),
        })

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for the SpeciesAssigner script.
    Loads metadata and shapefile, matches bbox centroids to polygons,
    and assigns class_ids to each bbox.
    """
    log.info("Starting SpeciesAssigner")
    try:
        assigner = SpeciesAssigner(cfg)
        assigner.run()
        log.info("Species assignment completed successfully.")
    except Exception as e:
        log.exception(f"SpeciesAssigner failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
