import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from omegaconf import DictConfig
import hydra
from shapely.geometry import Point
from tqdm import tqdm

log = logging.getLogger(__name__)

class SpeciesAssigner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.batch_id = cfg.batch_id
        self.season = cfg.season
        self.spec_dict = self.read_json(cfg.paths.species_info)
        self.metadata_path = Path(cfg.paths.batch_dir, "metadata")
        
        self.shapefile_path = Path(cfg.paths.semif_util_dir) / "autosfm" / "ShapeFiles" / self.season / f"{self.season}.shp"
        self.polygons = gpd.read_file(self.shapefile_path)
        self.closest_distance_thresh = 2  # meters

    def read_json(self, filepath: Path):
        with open(filepath) as f:
            metadata = json.load(f)
        return metadata
    
    def save_json(self, filepath: Path, data: dict):
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
    
    def run(self):
        start = time.time()
        metadata_files = sorted(self.metadata_path.glob("*.json"))
        for file in tqdm(metadata_files, desc="Assigning labels"):
            self._process_file(file)
        log.info(f"Assigning species completed in {time.time() - start:.2f} seconds.")

    def _process_file(self, filepath: Path):
        metadata = self.read_json(filepath)

        batch_id = metadata.get("batch_id", "")
        for bbox in metadata.get("annotations", []):
            species_info = self._determine_species(bbox, batch_id)
            self._assign_species(bbox, species_info)

        self.save_json(filepath, metadata)

    def _determine_species(self, bbox: dict, batch_id: str) -> dict:
        x, y = bbox["global_coordinates"]["global_centroid"]
        bbox_cls = bbox.get("category_class_id")
        if bbox_cls == 28:
            return self.spec_dict["species"].get("colorchecker", None)

        if "cash" in self.season and bbox_cls != 28:
            return self._get_cash_crop_species()

        point = Point(x, y)
        return self._lookup_species_from_point(point, bbox, batch_id)

    def _get_cash_crop_species(self) -> dict:
        batch_prefix = self.batch_id
        crop_lookup = {"NC": "GLMA4", "MD": "ZEA", "TX": "GOHI"}
        for prefix, species_code in crop_lookup.items():
            if prefix in batch_prefix:
                return self.spec_dict["species"].get(species_code, self.spec_dict["species"]["plant"])
        return self.spec_dict["species"]["plant"]

    def _lookup_species_from_point(self, point: Point, bbox: dict, batch_id: str) -> dict:
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

    def _handle_no_polygon_match(self, point: Point, bbox: dict) -> dict:
        log.warning(f"No polygon found for bbox_id: {bbox.get('bbox_id')}. Trying closest.")
        distances = self.polygons["geometry"].apply(lambda poly: poly.distance(point))
        closest_idx = distances.idxmin()
        closest_distance = distances[closest_idx]

        if closest_distance < self.closest_distance_thresh:
            closest_poly = self.polygons.loc[closest_idx]
            log.warning(f"Closest polygon used: {closest_poly['comm_name']}")
            return self.spec_dict["species"].get(closest_poly["species"], self.spec_dict["species"]["plant"])

        log.warning(f"No nearby polygon within {self.closest_distance_thresh}m. Using fallback.")
        return self.spec_dict["species"]["plant"]

    def _handle_undefined_species(self, containing, bbox, batch_id: str) -> dict:
        log.warning("Polygon has no defined species. Assigning as non_target_weed.")
        poly_id = containing["id"].values[0]

        if bbox.get("non_target_weed") != "non_target_weed":
            bbox["non_target_weed"] = "non_target_weed"
            bbox["non_target_weed_pred_conf"] = 1.0

        return self.spec_dict["species"]["plant"]

    def _assign_species(self, bbox: dict, species_info: dict):
        bbox.update({
            "category_class_id": species_info.get("class_id"),
        })

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # TODO: Save final format of metadata that matches schema
    # TODO: Add documentation and logging to all scripts
    assigner = SpeciesAssigner(cfg)
    assigner.run()

if __name__ == "__main__":
    main()