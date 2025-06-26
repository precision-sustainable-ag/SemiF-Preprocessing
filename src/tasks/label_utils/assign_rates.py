import json
import logging
from pathlib import Path
import geopandas as gpd
from omegaconf import DictConfig
import hydra
from shapely.geometry import Point, Polygon
from tqdm import tqdm

from src.utils.utils import safe_save_json

log = logging.getLogger(__name__)

class RateAssigner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.batch_id = cfg.batch_id
        self.season = cfg.season
        self.spec_dict = self.read_json(cfg.paths.species_info)
        self.metadata_path = Path(cfg.paths.batch_dir, "metadata")
        self.shapefile_path = Path(cfg.paths.semif_util_dir) / "autosfm" / "ShapeFiles" / self.season / f"{self.season}.shp"
        self.polygons = gpd.read_file(self.shapefile_path)
        self.bbox_shp_output_dir = Path(cfg.paths.inspection_dir) / "rate_bbox_shapefile"
        self.bbox_shp_output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Initialized RateAssigner for batch: {self.batch_id}, season: {self.season}")
        log.info(f"Loaded shapefile from: {self.shapefile_path}")

    def read_json(self, filepath: Path):
        with open(filepath) as f:
            metadata = json.load(f)
        return metadata
    
    def save_json(self, filepath: Path, data: dict):
        safe_save_json(data, filepath)
    
    def run(self) -> None:
        """
        Assign species labels to all bounding boxes found in image metadata.
        """
        metadata_files = sorted(self.metadata_path.glob("*.json"))
        log.info(f"Found {len(metadata_files)} metadata files to process.")
        for file in tqdm(metadata_files, desc="Assigning labels"):
            self._process_file(file)

    def _process_file(self, filepath: Path):
        """Load and process a single image metadata file."""
        metadata = self.read_json(filepath)

        batch_id = metadata.get("batch_id", "")
        for bbox in metadata.get("annotations", []):
            species_info, exp_info = self._determine_rate(bbox, batch_id)
            
            self._assign_species(bbox, species_info)
            self._assign_rate(bbox, exp_info)

        self.save_json(filepath, metadata)
        log.debug(f"Updated species labels in: {filepath.name}")

    def _determine_rate(self, bbox: dict, batch_id: str) -> dict:
        """
        Determine rate based on spatial location or fallback rules.
        """
        x, y = bbox["global_coordinates"]["global_centroid"]

        point = Point(x, y)
        cat_class_info, exp_info = self._lookup_rate_from_point(point, bbox, batch_id)

        return cat_class_info, exp_info


    def _lookup_rate_from_point(self, point: Point, bbox: dict, batch_id: str) -> dict:
        """Match bounding box centroid to polygon to infer rate."""
        contains_point = self.polygons["geometry"].apply(lambda poly: poly.contains(point))
        containing = self.polygons[contains_point]

        if len(containing):
            poly_row = containing.iloc[0]
        else:
            # Find the closest polygon by centroid (could also use .distance(point) for boundary)
            distances = self.polygons["geometry"].distance(point)
            min_idx = distances.idxmin()
            poly_row = self.polygons.loc[min_idx]
            log.warning(f"Point {point} not in any polygon for bbox {bbox.get('cutout_id', 'unknown')} in batch {batch_id}. Assigned to closest polygon with plot_id {poly_row['plot_id']}.")

        plant_id = poly_row["plant_id"]

        experimental_vals = {
            "dap_id": int(poly_row["dap_id"]),
            "block_id": int(poly_row["block_id"]),
            "rep_id": int(poly_row["rep_id"]),
            "row_id": int(poly_row["row_id"]),
            "plot_id": poly_row["plot_id"],
            "rate_gAE_A": float(poly_row["rate_gAE_A"]),
        }
        
        cat_class_info = self.spec_dict["species"].get(plant_id, self.spec_dict["species"]["plant"])
        
        return cat_class_info, experimental_vals


    def _assign_species(self, bbox: dict, species_info: dict):
        bbox.update({
            "category_class_id": species_info.get("class_id"),
        })

    def _assign_rate(self, bbox: dict, rate_info: dict):
        """
        Assign rate information to the bounding box.
        """
        bbox.update({"experiment_info": rate_info})

    def export_bboxes_to_shapefile(self):
        """
        Export all bounding boxes (using global_coordinates) as polygons to a shapefile with experiment info.
        """
        all_records = []
        metadata_files = sorted(self.metadata_path.glob("*.json"))
        for file in tqdm(metadata_files, desc="Collecting bboxes for shapefile"):
            metadata = self.read_json(file)
            for bbox in metadata.get("annotations", []):
                global_coords = bbox.get("global_coordinates", {})
                try:
                    # Build the polygon in order: top_left, top_right, bottom_right, bottom_left
                    corners = [
                        tuple(global_coords["top_left"]),
                        tuple(global_coords["top_right"]),
                        tuple(global_coords["bottom_right"]),
                        tuple(global_coords["bottom_left"]),
                    ]
                    # Optionally close the polygon (shapely does this automatically, but explicit is fine)
                    poly = Polygon(corners)
                except Exception as e:
                    log.warning(f"Skipping bbox with missing corners in {file.name}: {e}")
                    continue

                exp_info = bbox.get("experiment_info", {})
                area_sqm = bbox.get("global_coordinates", {}).get("area_sqm", 0)
                area_sqcm = area_sqm * 10000  # 1 sqm = 10,000 sqcm
                record = {
                    "geometry": poly,
                    "cutout_id": bbox.get("cutout_id", ""),
                    "class_id": bbox.get("category_class_id", None),
                    "is_primary": bbox.get("is_primary", None),
                    "area_sqcm": area_sqcm,
                    **exp_info,
                }
                all_records.append(record)

        if not all_records:
            log.warning("No bounding boxes found to export.")
            return

        gdf = gpd.GeoDataFrame(all_records, crs=self.polygons.crs if hasattr(self.polygons, "crs") else None)
        output_path = self.bbox_shp_output_dir / f"{self.batch_id}_rate_bboxes.shp"
        gdf.to_file(output_path)
        log.info(f"Exported {len(gdf)} bounding boxes to {output_path}")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for the RateAssigner script.
    Loads metadata and shapefile, matches bbox centroids to polygons,
    and assigns class_ids to each bbox.
    """
    log.info("Starting RateAssigner")
    try:
        assigner = RateAssigner(cfg)
        assigner.run()
        
        log.info("Rate assignment completed successfully.")
        assigner.export_bboxes_to_shapefile()
    except Exception as e:
        log.exception(f"RateAssigner failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
