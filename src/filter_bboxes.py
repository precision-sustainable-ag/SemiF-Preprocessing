import logging
from typing import Dict, List, Optional
import numpy as np

import json
from pathlib import Path
from omegaconf import DictConfig
import time
from shapely.geometry import Polygon
from tqdm import tqdm

log = logging.getLogger(__name__)

FOV_IOU_THRESH = 0.1
BBOX_OVERLAP_THRESH = 0.3

def generate_hash(box: Dict, auxiliary_hash: Optional[str] = None) -> str:
    box_id = box.get("bbox_id") or box.get("id")
    if not box_id:
        raise ValueError("Box is missing 'bbox_id' or 'id'")
    box_hash = str(box_id)
    if auxiliary_hash is not None:
        box_hash = ",".join(sorted([auxiliary_hash, box_hash]))
    return box_hash

class BBoxFilter:
    def __init__(self, cfg, images: List[dict]=None, load_existing: bool = False):
        
        self.batch_dir = Path(cfg.paths.batch_dir)
        self.metadata_output_dir = self.batch_dir / "metadata"

        if load_existing:
            self.images = self.load_existing_metadata()
        else:
            self.images = images

        self.image_map = {image["image_id"]: image for image in self.images}
        self.total_bboxes = sum([len(image["bboxes"]) for image in self.images])
        self.primary_boxes = []
        self.primary_box_ids = set()

    def load_existing_metadata(self) -> List[dict]:
        """
        Loads previously saved JSON metadata if available.
        """
        json_files = sorted(self.metadata_output_dir.glob("*.json"))
        image_data = []

        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)
                image_data.append(data)
        
        log.info(f"Loaded {len(image_data)} image metadata files from JSON.")
        return image_data

    def deduplicate_bboxes(self):
        """Calculates the ideal bounding box and the associated image from all the
        bounding boxes
        """
        comparisons = self._filter_images_by_fov()
        # Comment out cleanup step for debug
        self.filter_bounding_boxes(comparisons)
        self.select_best_bbox()
        self.cleanup_primary_boxes()
        self.cleanup_overlapping_bboxes()

    def _filter_images_by_fov(self) -> Dict[str, List[str]]:
        """Filter the images to compare based on the overlap between their fields of
           view

        Returns:
            Dict[str, List[str]]: A dictionary containing the image IDs as keys, and
                                  a list of image IDs each key overlaps with
        """
        image_ids = list(self.image_map.keys())
        comparisons = dict()
        # Find the overlap between FOVs of the images
        for i, image_id in enumerate(image_ids):
            image = self.image_map[image_id]
            comparisons[image_id] = []
            for j in range(i + 1, len(image_ids)):
                compare_image_id = image_ids[j]
                compare_image = self.image_map[compare_image_id]
                # fov_iou = bb_iou(image["camera_info"]["fov"], compare_image["camera_info"]["fov"])
                fov_iou = self._simple_bb_iou(image["camera_info"]["fov"], compare_image["camera_info"]["fov"])
                if fov_iou > FOV_IOU_THRESH:
                    comparisons[image_id].append(compare_image_id)

        return comparisons


    def filter_bounding_boxes(self, comparisons: Dict[str, List[str]]):
        """Find overlapping bounding boxes from the images to compare

        Args:
            comparisons (Dict[str, List[str]]): Images to compare, found via
                                                filter_images
        """
        # For all the overlapping images
        visited_bboxes = set()

        for image_id, compare_ids in tqdm(comparisons.items()):
            # For each bounding box in the key image
            for box in self.image_map[image_id]["bboxes"]:
                box.setdefault("_overlapping_bboxes", [])
                box["is_primary"] = False
                if box["bbox_id"] in visited_bboxes:
                    continue

                visited_bboxes.add(box["bbox_id"])
                compared = set()
                box_hash = generate_hash(box)

                # For each overlapping image
                for compare_image_id in compare_ids:
                    for other_box in self.image_map[compare_image_id]["bboxes"]:
                        other_box.setdefault("_overlapping_bboxes", [])
                        other_box["is_primary"] = False
                        if other_box["bbox_id"] in visited_bboxes:
                            continue
                    
                        # A unique ID for a pair of bounding boxes
                        # Note that the order of the boxes does not matter
                        # i.e. Box_A,Box_B is the same as Box_B,Box_A
                        other_hash = generate_hash(other_box, box_hash)
                        if other_hash in compared:
                            continue
                        
                        compared.add(other_hash)
                        iou = self._precise_bb_iou(box, other_box)
                        
                        if iou > BBOX_OVERLAP_THRESH:
                            box["_overlapping_bboxes"].append(other_box["bbox_id"])
                            other_box["_overlapping_bboxes"].append(box["bbox_id"])
                            visited_bboxes.add(other_box["bbox_id"])
                            

        

    def select_best_bbox(self):
        # visited will be a set of boxes that have been compared
        visited = set()
        for image in self.images:
            # If all the boxes have been checked, no need to
            # check the other images
            if len(visited) == self.total_bboxes:
                break

            for box in image["bboxes"]:
                box_hash = generate_hash(box)
                if box_hash in visited:
                    continue


                all_boxes = [box] + [self._get_box_by_id(bid) for bid in box.get("_overlapping_bboxes", [])]
                box_hashes = [generate_hash(b) for b in all_boxes]
                visited.update(box_hashes)

                for b in all_boxes:
                    b["is_primary"] = False

                # Find the best bounding box
                centers = np.array([self.image_map[b["image_id"]]["camera_info"]["camera_location"] for b in all_boxes])
                centroids = np.array([b["global_coordinates"]["global_centroid"] for b in all_boxes])
                
                distances = 0

                try:
                    distances = ((centroids - centers[:, :2]) ** 2).sum(axis=-1)
                except ValueError as e:
                    log.exception(f"Error calculating distances: {str(e)}")
                    log.error(f"Centroids: {centroids}")
                    log.error(f"Centers: {centers}")
                    log.error(f"Centers [:, :2]: {centers[:, :2]}")
                    continue

                best_idx = np.argmin(distances)
                best_box = all_boxes[best_idx]
                best_box["is_primary"] = True

                log.info(f"Selected primary bbox: {best_box['bbox_id']} from image {best_box['image_id']}")


                if best_box["bbox_id"] not in self.primary_box_ids:
                    self.primary_boxes.append(best_box)
                    self.primary_box_ids.add(best_box["bbox_id"])

    def cleanup_overlapping_bboxes(self):
        """Remove duplicate entries in the _overlapping_bboxes field and sort them"""
        for image in self.images:
            for box in image["bboxes"]:
                box["_overlapping_bboxes"] = list(set(box["_overlapping_bboxes"]))
                box["_overlapping_bboxes"].sort()


    def cleanup_primary_boxes(self):
        _primary_boxes = []
        for box in self.primary_boxes:
            image = self.image_map[box["image_id"]]
            w, h = image["width"], image["height"]
            x_norm, y_norm = box["local_coordinates"]["local_centroid"]
            x = x_norm * w
            y = y_norm * h
        
            if w // 4 < x < 3 * w // 4 and h // 4 < y < 3 * h // 4:
                _primary_boxes.append(box)
        
            else:
                box["is_primary"] = False

        # Revisit all bounding boxes identified as primary and
        # remove the overlapping ones
        for i, box1 in enumerate(_primary_boxes):
            cam1 = np.array(self.image_map[box1["image_id"]]["camera_info"]["camera_location"][:2])
            for j in range(i + 1, len(_primary_boxes)):
                box2 = _primary_boxes[j]
                cam2 = np.array(self.image_map[box2["image_id"]]["camera_info"]["camera_location"][:2])
                
                iou = self._precise_bb_iou(box1, box2)

                if iou > BBOX_OVERLAP_THRESH:
                    dist1 = np.sum((np.array(box1["global_coordinates"]["global_centroid"]) - cam1) ** 2)
                    dist2 = np.sum((np.array(box2["global_coordinates"]["global_centroid"]) - cam2) ** 2)
                    if dist1 < dist2:
                        box2["is_primary"] = False
                    else:
                        box1["is_primary"] = False
    
    def _simple_bb_iou(self, boxA: dict, boxB: dict) -> float:
        a = [boxA["top_left"], boxA["bottom_right"]]
        b = [boxB["top_left"], boxB["bottom_right"]]
        xA = max(a[0][0], b[0][0])
        yA = max(-a[0][1], -b[0][1])
        xB = min(a[1][0], b[1][0])
        yB = min(-a[1][1], -b[1][1])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        if inter_area == 0:
            return 0.0
        areaA = abs((a[1][0] - a[0][0]) * (a[1][1] - a[0][1]))
        areaB = abs((b[1][0] - b[0][0]) * (b[1][1] - b[0][1]))
        return inter_area / (areaA + areaB - inter_area)

    def _precise_bb_iou(self, boxA: dict, boxB: dict, coord_type: str = "global") -> float:
        try:
            coordsA = boxA[f"{coord_type}_coordinates"]
            coordsB = boxB[f"{coord_type}_coordinates"]
            polyA = Polygon([coordsA["top_left"], coordsA["top_right"], coordsA["bottom_right"], coordsA["bottom_left"]])
            polyB = Polygon([coordsB["top_left"], coordsB["top_right"], coordsB["bottom_right"], coordsB["bottom_left"]])
            if not polyA.is_valid:
                polyA = polyA.buffer(0)
            if not polyB.is_valid:
                polyB = polyB.buffer(0)
            if not polyA.intersects(polyB):
                return 0.0
            return polyA.intersection(polyB).area / polyA.union(polyB).area
        except Exception as e:
            log.error(f"Failed to compute precise IoU: {e}")
            return 0.0
    
    def _get_box_by_id(self, bbox_id: str) -> Optional[dict]:
        for image in self.images:
            for box in image["bboxes"]:
                if box["bbox_id"] == bbox_id:
                    return box
        return None


def main(cfg: DictConfig) -> None:
    # TODO documentation
    start = time.time()
    bbox_filter = BBoxFilter(cfg)
    bbox_filter.deduplicate_bboxes()
    imgs = bbox_filter.images
    Path(bbox_filter.metadata_output_dir).mkdir(parents=True, exist_ok=True)

    for img in imgs:
        with open(bbox_filter.metadata_output_dir / f"{img['image_id']}.json", "w") as f:
            json.dump(img, f, indent=4)
        
    
    end = time.time()
    log.info(f"Bbox filtering completed in {end - start} seconds.")

if __name__ == "__main__":
    main()
