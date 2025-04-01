from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import json


# read json file
def read_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

batch_dir = Path("data/longterm_images2/semifield-developed-images/NC_2025-03-17")
metadata_dir = batch_dir / "metadata"
image_dir = batch_dir / "autosfm" / "downscaled_photos"

# ensure output directory exists
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

images = list(image_dir.glob("*.jpg"))

# Load the image, read the associated metadata file, and display the bboxes on the images and save to a file
for image in images:
    image_name = image.name
    image_stem = image.stem
    metadata_path = metadata_dir / f"{image_stem}.json"
    image_path = str(image)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img)

    metadata = read_json(metadata_path)
    
    for bbox in metadata["bboxes"]:
        local_coordinates = bbox["local_coordinates"]
        xmin = local_coordinates["top_left_pixel"][0]
        ymin = local_coordinates["top_left_pixel"][1]
        xmax = local_coordinates["bottom_right_pixel"][0]
        ymax = local_coordinates["bottom_right_pixel"][1]

        global_coordinates = bbox["global_coordinates"]
        g_xmin = round(global_coordinates["top_left"][0], 8)
        g_ymin = round(global_coordinates["top_left"][1], 8)

        w = xmax - xmin
        h = ymax - ymin

        
        rect = plt.Rectangle((xmin, ymin), w, h, fill=False, color="red")
        ax.add_patch(rect)
        
        # Add coordinate text
        coord_text = f"({g_xmin}, {g_ymin})"
        ax.text(
            xmin, ymin - 5,
            coord_text,
            fontsize=8,
            color="yellow",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.2")
        )
    
    plt.axis("off")
    plt.savefig(f"output/{image_name}", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved image with bboxes to output/{image_name}")
