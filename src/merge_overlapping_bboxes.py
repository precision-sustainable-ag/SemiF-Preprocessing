import pandas as pd
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
import hydra
import logging
import networkx as nx

log = logging.getLogger(__name__)



def merge_bboxes_with_class(bboxes, iou_threshold=0.5):
    """
    Merge bounding boxes while carrying the class along.
    Input:
      - bboxes: a list of dictionaries. Each dictionary must have:
          'xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class', 'classname'
      - iou_threshold: threshold for considering boxes overlapping.
      
    For each merged group:
      - The coordinates are merged (taking the min and max over the group).
      - If any box in the group is a "colorchecker", then the merged box is marked as a colorchecker.
      - Otherwise, it remains a "plant".
    """
    n = len(bboxes)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Build graph: add an edge if boxes overlap or one is contained in the other.
    for i in range(n):
        for j in range(i + 1, n):
            box_i = [bboxes[i]['xmin'], bboxes[i]['ymin'], bboxes[i]['xmax'], bboxes[i]['ymax']]
            box_j = [bboxes[j]['xmin'], bboxes[j]['ymin'], bboxes[j]['xmax'], bboxes[j]['ymax']]
            if (iou(box_i, box_j) >= iou_threshold or 
                is_contained(box_i, box_j) or 
                is_contained(box_j, box_i)):
                G.add_edge(i, j)

    merged_boxes = []
    # Process each connected component (group of boxes to merge)
    for component in nx.connected_components(G):
        comp_boxes = [bboxes[i] for i in component]
        # Merge the coordinates
        xmin = min(b['xmin'] for b in comp_boxes)
        ymin = min(b['ymin'] for b in comp_boxes)
        xmax = max(b['xmax'] for b in comp_boxes)
        ymax = max(b['ymax'] for b in comp_boxes)
        # Decide on class: if any box is "colorchecker", mark as such.
        classes = [b['classname'] for b in comp_boxes]
        if "colorchecker" in classes:
            classname = "colorchecker"
            cls = 1  # assuming numeric class 1 = colorchecker
        else:
            classname = "plant"
            cls = 0

        # For confidence, you might choose max, average, etc. Here we use max.
        conf = max(b['conf'] for b in comp_boxes)
        merged_box = {
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'conf': conf,
            'class': cls,
            'classname': classname
        }
        merged_boxes.append(merged_box)
    return merged_boxes


def iou(box1, box2):
    """
    Compute Intersection over Union for two bounding boxes.
    Each box is in the format: [xmin, ymin, xmax, ymax]
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    # Calculate intersection coordinates
    xi1 = max(xmin1, xmin2)
    yi1 = max(ymin1, ymin2)
    xi2 = min(xmax1, xmax2)
    yi2 = min(ymax1, ymax2)

    # Calculate the area of the intersection rectangle
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate the areas of both bounding boxes
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

    # Calculate IoU
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def is_contained(box1, box2):
    """Check if box2 is fully contained within box1."""
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    return (xmin1 <= xmin2 <= xmax2 <= xmax1) and (ymin1 <= ymin2 <= ymax2 <= ymax1)

def merge_boxes(boxes, threshold=0.5):
    """Merge overlapping or fully contained bounding boxes into a single larger one."""
    merged_boxes = []

    while boxes:
        # Take the first box
        current_box = boxes.pop(0)
        xmin, ymin, xmax, ymax = current_box

        to_merge = [current_box]
        remaining_boxes = []

        # Check for overlap or containment with remaining boxes
        for box in boxes:
            if iou(current_box, box) >= threshold or is_contained(current_box, box) or is_contained(box, current_box):
                # Merge with overlapping or contained box by updating boundaries
                xmin = min(xmin, box[0])
                ymin = min(ymin, box[1])
                xmax = max(xmax, box[2])
                ymax = max(ymax, box[3])
                to_merge.append(box)
            else:
                remaining_boxes.append(box)

        # Add the new merged box to the result list
        new_box = [xmin, ymin, xmax, ymax]
        merged_boxes.append(new_box)

        # Continue with the remaining boxes
        boxes = remaining_boxes

    return merged_boxes


def process_csv_file(csv_path, output_dir, iou_threshold=0.5):
    """
    Read a CSV file with bounding boxes, merge overlapping boxes while carrying
    the class information (only two classes: plant and colorchecker), and write back.
    
    The CSV is expected to have the columns:
      bounding_box_id, xmin, ymin, xmax, ymax, conf, class, classname
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        log.error(f"Error reading CSV {csv_path}: {e}")
        return

    # Check required columns
    required_cols = ['bounding_box_id', 'xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class', 'classname']
    for col in required_cols:
        if col not in df.columns:
            log.error(f"CSV {csv_path} is missing required column '{col}'")
            return

    # Ensure numeric columns are numbers
    for col in ['xmin', 'ymin', 'xmax', 'ymax', 'conf']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['xmin', 'ymin', 'xmax', 'ymax'])

    # Convert the DataFrame to a list of dictionaries (each representing one bbox)
    bboxes = df.to_dict(orient='records')
    if not bboxes:
        log.warning(f"No valid bounding boxes found in {csv_path}")
        return

    merged_bboxes = merge_bboxes_with_class(bboxes, iou_threshold=iou_threshold)
    
    # If merging changed the number of boxes (i.e. some were merged together),
    # we keep only the class information (as carried by our merging function).
    merged_df = pd.DataFrame(merged_bboxes)

    # Optionally, you can reassign new bounding_box_id values.
    merged_df.insert(0, 'bounding_box_id', range(len(merged_df)))
    
    try:
        csv_path = output_dir / csv_path.name 
        merged_df.to_csv(csv_path, index=False)
        log.info(f"Processed and saved merged boxes to {csv_path}")
    except Exception as e:
        log.error(f"Error writing CSV {csv_path}: {e}")

def process_all_csvs_in_directory(directory_path, output_dir, iou_threshold=0.5):
    """Process all CSV files in the given directory."""
    directory = Path(directory_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files in the directory
    csv_files = list(directory.rglob("*.csv"))
    log.info(f"Found {len(csv_files)} CSV files to process.")

    # Process each CSV file
    for csv_file in csv_files:
        process_csv_file(csv_file, output_dir, iou_threshold)

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to process CSVs in a directory."""
    csv_directory = Path(cfg.paths.batch_dir) / "plant-detections"  # Directory containing CSV files
    output_dir = csv_directory / "merged"  # Directory for saving merged CSVs
    iou_threshold = 0.5  # Default IoU threshold for merging

    # Process all CSVs in the specified directory
    process_all_csvs_in_directory(csv_directory,output_dir, iou_threshold)

if __name__ == "__main__":
    main()
    