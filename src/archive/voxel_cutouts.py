from pathlib import Path
import fiftyone as fo
import json
from typing import List, Dict, Any, Union

def read_json(json_file: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    with open(json_file, 'r') as f:
        return json.load(f)

def get_batches(root_dir: Union[str, Path]) -> List[Path]:
    """Return sorted list of batch directories in root_dir."""
    return sorted(Path(root_dir).glob("*"))

def get_cutouts(batch_dir: Path) -> List[Path]:
    """Return sorted list of .jpg cutout files in a batch directory."""
    return sorted(batch_dir.glob("*.png"))


if __name__ == "__main__":

    images_patt = "voxel_test_data/semifield-cutouts"

    batches = get_batches(images_patt)

    samples = []
    for batch in batches:
        cutouts = get_cutouts(batch)
        for cutout_path in cutouts:
            mask_path = cutout_path.parent / f"{cutout_path.stem}_mask.png"
            mask_path = mask_path.resolve()

            metadata_path = cutout_path.with_suffix(".json")
            try:
                metadata = read_json(metadata_path)
            except Exception as e:
                print(f"Could not read metadata {metadata_path}: {e}")
                continue

            cutout_props = metadata.get("cutout_props", {})
            non_target_weed = cutout_props.get("non_target_weed")
            
            # Voxel only excepts True or False, so convert to string to be able to include "None"
            if non_target_weed is True:
                non_target_weed_str = "True"
            elif non_target_weed is False:
                non_target_weed_str = "False"
            else:
                non_target_weed_str = "None"

            class_id = metadata.get("category", {}).get("class_id")
            categories = metadata.get("categories", {}) # this is a list
            common_name = None
            common_name = next((cat.get("common_name") for cat in categories if cat.get("class_id") == class_id), None)


            sample = fo.Sample(
                filepath=cutout_path,
                ground_truth=fo.Segmentation(mask_path=str(mask_path)),
                batch_id=metadata.get("batch_id"),
                is_primary=cutout_props.get("is_primary"),
                extends_border=cutout_props.get("extends_border"),
                bbox_area_cm2=cutout_props.get("bbox_area_cm2"),
                blur_effect=cutout_props.get("blur_effect"),
                num_components=cutout_props.get("num_components"),
                non_target_weed=non_target_weed_str,
                common_name=common_name,
            )

            samples.append(sample)

    # Create or overwrite the dataset
    dataset = fo.Dataset("semifield-cutouts")
    dataset.add_samples(samples)

    session = fo.launch_app(dataset)
    session.wait()
