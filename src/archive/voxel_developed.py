from pathlib import Path
import fiftyone as fo
import json
import math
from typing import List, Dict, Any, Union

def is_valid_bounding_box(bbox: List[float]) -> bool:
    """Returns True if bbox contains all valid (not None/NaN) floats."""
    return all(
        isinstance(v, float) and v is not None and not math.isnan(v)
        for v in bbox
    )

def read_json(json_file: Union[str, Path]) -> Dict[str, Any]:
    """Reads a JSON file and returns its data."""
    with open(json_file, "r") as f:
        return json.load(f)

def get_batches(root_dir: Union[str, Path]) -> List[Path]:
    """Returns a sorted list of batch directories in the root directory."""
    return sorted(Path(root_dir).glob("*"))

def get_images(images_dir: Union[str, Path]) -> List[Path]:
    """Returns a sorted list of all .jpg image files in the directory."""
    return sorted(Path(images_dir).glob("*.jpg"))

if __name__ == "__main__":
    root_dir = "voxel_test_data/semifield-developed-images"
    batches = get_batches(root_dir)

    samples = []

    for batch in batches:
        images_dir = batch / "images"
        images = get_images(images_dir)
        for image_path in images:

            

            metadata_path = image_path.parent.parent / "metadata" / (image_path.stem + ".json")
            mask_path = image_path.parent.parent / "masks/semantic_masks" / (image_path.stem + ".png")
            sample = fo.Sample(filepath=image_path,
                               ground_truth=fo.Segmentation(mask_path=str(mask_path)))
            try:
                metadata = read_json(metadata_path)
            except Exception as e:
                print(f"Could not read metadata {metadata_path}: {e}")
                continue

            batch_id = metadata.get("batch_id")
            exif_meta = metadata.get("exif_meta", {})
            h, w = exif_meta.get("ImageLength"), exif_meta.get("ImageWidth")
            if not h or not w:
                print(f"Skipping image {image_path} due to missing dimensions.")
                continue

            detections = []
            annotations = metadata.get("annotations", [])
            for idx, annotation in enumerate(annotations):
                class_id = annotation.get("category_class_id")

                categories = metadata.get("categories", {}) # this is a list

                common_name = next((cat.get("common_name") for cat in categories if cat.get("class_id") == class_id), None)

                bbox = annotation.get("bbox_xywh")
                if not bbox or len(bbox) != 4:
                    print(f"Annotation missing bbox for {image_path}: {annotation}")
                    continue

                x, y, bboxw, bboxh = bbox
                norm_x, norm_y = x / w, y / h
                norm_bboxw, norm_bboxh = bboxw / w, bboxh / h

                bounding_box = [norm_x, norm_y, norm_bboxw, norm_bboxh]
                if is_valid_bounding_box(bounding_box):
                    detection = fo.Detection(
                        common_name=common_name,
                        bounding_box=bounding_box,
                        index=idx,
                    )
                    detections.append(detection)
                else:
                    print(f"Invalid bounding box for {image_path}: {bounding_box}")

            sample["ground_truth"] = fo.Detections(detections=detections)
            samples.append(sample)

    # Create or overwrite the dataset
    dataset = fo.Dataset("semifield-developed-images")
    dataset.add_samples(samples)

    session = fo.launch_app(dataset)
    session.wait()
