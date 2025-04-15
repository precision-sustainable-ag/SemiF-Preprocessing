import datetime
import getpass
import json
import logging
import random
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import hydra
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.utils import find_lts_dir
log = logging.getLogger(__name__)

random.seed(42)  # For reproducibility
GITHUB_REPO_URL = "https://github.com/precision-sustainable-ag/SemiF-Preprocessing/issues"

LABEL_OPTIONS = {
    "1": "Pass",
    "2": "Preprocessing Quality",
    "3": "Potting Area Cleanliness",
    "4": "Non-Target",
    "5": "Plant Spacing",
    "6": "Incorrect Species",
    "7": "Bad size estimate",
    "0": "Other",
    "q": "Quit",
    "b": "Back"
}
class AnnotationPlotter:
    """
    A class for loading annotation metadata and generating summary plots
    such as species counts, area histograms, and centroid density heatmaps.
    """
    def __init__(self, cfg: DictConfig):
        self.metadata_dir = Path(cfg.paths.batch_dir) / "metadata"
        self.save_dir = Path(cfg.paths.inspection_dir) / "plots"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(cfg.paths.species_info, 'r') as f:
            species_data = json.load(f)
        
        # remap species class_id to common name
        self.species_info = {
            str(species["class_id"]): species["common_name"]
            for species in species_data["species"].values()
        }

    def load_annotation_data(self) -> pd.DataFrame:
        """
        Loads annotation data from JSON files in the metadata directory.

        Returns:
            pd.DataFrame: A DataFrame with species ID, area (cm²), bounding box, and centroid coordinates.
        """
        records = []
        for json_file in sorted(self.metadata_dir.glob("*.json")):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    annotations = data.get("annotations", [])
                    for ann in annotations:
                        species_id = ann.get("category_class_id")
                        species_id = self.species_info[str(species_id)].lower()
                        x, y, w, h = ann.get("bbox_xywh", [None]*4)
                        centroid = ann.get("local_coordinates", {}).get("local_centroid")
                        x_centroid, y_centroid  = centroid[0], centroid[1]
                        area = ann.get("global_coordinates", {}).get("area_sqm") * 10000  # Convert to square cm
                        if species_id is not None and area is not None:
                            records.append({"species_id": species_id, "area_sqcm": area, "x": x, "y": y, "w": w, "h": h, "x_centroid": x_centroid, "y_centroid": y_centroid})
            except Exception as e:
                log.error(f"Failed to process {json_file.name}: {e}", exc_info=True)
        return pd.DataFrame(records)

    def plot_species_centroid_density(self, df: pd.DataFrame):
        """
        Creates density heatmaps of centroid positions per species.
        """
    
        plt.figure(figsize=(10, 8))

        g = sns.FacetGrid(df, col="species_id", col_wrap=3, height=3.5)
        g.map_dataframe(sns.kdeplot, x="x_centroid", y="y_centroid", fill=True, cmap="viridis", bw_adjust=0.5, clip=((0, 1), (0, 1)))
        g.set_titles("{col_name}")
        g.set_axis_labels("X (relative)", "Y (relative)")
        for ax in g.axes.flatten():
            ax.invert_yaxis()

        # Add a shared colorbar (density scale)
        norm = colors.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar_ax = g.figure.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
        cbar = g.figure.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Relative Density")

        g.figure.suptitle("Centroid Density per Species (Normalized)", y=1.02)
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
        plot_path = self.save_dir / "species_centroid_density.png"
        plt.savefig(plot_path, dpi=300)
        log.info(f"Centroid density plot saved to {plot_path}")

    def log_scale_histogram(self, df: pd.DataFrame, bins: int = 30):
        """
        Creates log-scaled histograms of plant area (in cm²) for each species.
        """
        # Plot 2: Log-scaled histogram per species
        g = sns.FacetGrid(df, col="species_id", col_wrap=3, sharey=False, height=3.5)
        g.map_dataframe(sns.histplot, x="area_sqcm", bins=bins, log_scale=(True, False))
        g.set_titles("{col_name}")
        g.set_axis_labels("Area (cm², log scale)", "Count")
        g.figure.suptitle("Log-Scaled Histograms of Area per Species")
        plt.tight_layout()
        plot_path = self.save_dir / "area_log_scaled_histograms.png"
        plt.savefig(plot_path, dpi=300)
        log.info(f"[PLOT_SAVED] Area histogram plot saved to {plot_path}")

    def species_count(self, df: pd.DataFrame):
        """
        Creates a bar plot showing the number of annotations per species.
        """
        species_counts = df["species_id"].value_counts().sort_index()
        plt.figure(figsize=(8, 5))
        ax = species_counts.plot(kind="bar")
        plt.title("Number of Annotations per Species")
        plt.xlabel("Species ID")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.grid(axis="y")

        # Add values on top of each bar
        for p in ax.patches:
            ax.annotate(
            str(p.get_height()),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=10
            )

        plot_path = self.save_dir / "species_counts.png"
        plt.savefig(plot_path, dpi=300)
        log.info(f"[PLOT_SAVED] Species count plot saved to {plot_path}")

    def plot_summary(self) -> None:
        """
        Generates all annotation summary plots including:
        - Species count
        - Area histogram
        - Centroid density
        """
        df = self.load_annotation_data()
        if df.empty:
            log.warning("No annotation data found.")
            return

        self.species_count(df)
        self.log_scale_histogram(df)
        self.plot_species_centroid_density(df)

class ImageReviewer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.batch_id = cfg.batch_id
        self.use_lts_images = cfg.inspection.use_lts_images
        
        self.lts_locations = cfg.paths.lts_locations
        self.lts_dir = find_lts_dir(self.batch_id, self.lts_locations, local=False, developed=True, dngs=False, jpgs=True)
        self.lts_dir_name = Path(self.lts_dir).name
        self.lts_batch_dir = Path(self.lts_dir) / "semifield-developed-images" / self.batch_id
        
        self.batch_folder = Path(cfg.paths.batch_dir)

        # Inputs
        self.image_dir = Path(cfg.paths.down_photos)
        self.metadata_dir = Path(cfg.paths.batch_dir) / "metadata"
        self.species_info = self.read_species_info(Path(cfg.paths.species_info))
        
        # Outputs        
        self.inspection_dir = self.lts_batch_dir / "inspection" if self.use_lts_images else Path(cfg.paths.inspection_dir)
        self.csv_file = self.inspection_dir / f"{self.batch_id}_label_inspection.csv"
        self.remapped_sample_dir = self.inspection_dir / "remapped_samples"
        if not self.remapped_sample_dir.exists():
            self.remapped_sample_dir.mkdir(parents=True, exist_ok=True)

        self.sample_size = 75
        self.images = self._get_image_paths()
        self.results = self._load_existing_results()
        
        # Get the current timestamp and user
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.user = getpass.getuser()

    def read_species_info(self, path: Path) -> dict[str, str]:
        """
        Reads species info JSON and returns mapping from class ID to common name.
        """
        if not path.exists():
            log.error(f"Species info file not found: {path}")
            return None
        with open(path, 'r') as f:
            species_data = json.load(f)
        # remap species class_id to common name
        species_data = {
            str(species["class_id"]): species["common_name"]
            for species in species_data["species"].values()
        }
        return species_data
    
    def _find_multiple_species_images(self) -> list[str]:
        """
        Identifies images with multiple species in their annotations.

        Returns:
            List[str]: Image IDs with more than one species.
        """
        multiple_species_images = []
        for file in self.metadata_dir.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                class_ids = {str(a.get("category_class_id", "Unknown")) for a in data.get("annotations", [])}
                if len(class_ids) > 1:
                    multiple_species_images.append(data["image_id"])
            except Exception as e:
                log.error(f"Failed finding multiple species: {file.name}: {e}", exc_info=True)
        return multiple_species_images
            
    

    def _get_image_paths(self) -> list[Path]:
        """Load images and return a sorted list of (a subset of) unlabeled ones."""
        images = sorted(self.image_dir.glob("*.jpg"))
        # Find images with multiple species. These help confirm species group separation is accurate.
        multi_species_images = self._find_multiple_species_images()
        multi_spec_img_paths = [self.image_dir / f"{img_id}.jpg" for img_id in multi_species_images]
        # Get a random sample of images
        random_sample = random.sample(images, min(len(images), self.sample_size))
        # Combine the random sample with the multi-species images
        data_paths = sorted(random_sample + multi_spec_img_paths)
        return data_paths

    def display_instructions(self) -> None:
        """Prints instructions for user input."""
        print("\n--- Image Quality Assessment ---")
        for key, label in LABEL_OPTIONS.items():
            display_key = f"{key} (zero)" if key == "0" else key
            bright_key = f"\033[1;97m{display_key}\033[0m"
            print(f"{bright_key} - {label}")
        print("\n🔄 Please wait while the X11 or X410 forwarding initializes. This may take a few seconds...\n")

    def confirm_save_results(self) -> None:
        """Ask the user if they want to save the final CSV. If not, delete the file."""
        while True:
            confirm = input("\n💾 Do you want to save the final inspection results? (y/n): ").strip().lower()
            if confirm == "y":
                self._save_results()
                log.info(f"Inspection results saved to {self.csv_file}")
                return self.csv_file
            elif confirm == "n":
                if self.csv_file.exists():
                    self.csv_file.unlink()
                    log.info(f"Inspection results discarded. {self.csv_file} removed.")
                else:
                    log.warning("No saved CSV file found to delete.")
                return None
            else:
                print("⚠️ Invalid input. Please enter 'y' to save or 'n' to discard.")

    def _load_existing_results(self) -> list[list[str]]:
        """Load existing CSV results or return an empty list."""
        if self.csv_file.exists():
            log.info(f"Loading existing results from {self.csv_file}")
            return pd.read_csv(self.csv_file).values.tolist()
        return []

    def _save_results(self) -> None:
        """Save the labeling results to a CSV file."""
        df = pd.DataFrame(
            self.results,
            columns=['BatchID', 'ImageID', 'Selection', 'Timestamp', 'User', 'LTSLocation']
        )
        df.to_csv(self.csv_file, index=False)

    def review_images(self):
        """Iterate over images and allow the user to label them."""
        sample_images = sorted(list(self.remapped_sample_dir.glob("*.jpg")))
        if not sample_images:
            log.warning("No sample images found for review.")
            return None

        self.display_instructions()
        cv2.namedWindow("Inspection Viewer")
        index = 0
        while index < len(sample_images):
            img_path = sample_images[index]
            if not self._display_bboxes_on_image(img_path):
                index += 1
                continue

            label = self._get_user_input()
            if label == "Quit":
                print("\n❌ Exiting image review.")
                cv2.destroyAllWindows()
                return self.csv_file  # Save progress and exit

            if label == "Back":
                if index > 0:
                    print("\n🔙 Going back to the previous image.")
                    self.results.pop()  # Remove last entry
                    index -= 1
                else:
                    print("⚠️ Already at the first image, cannot go back further.")
                continue

            self.results.append([self.batch_id, img_path.stem, label, self.timestamp, self.user, self.lts_dir_name])
            self._save_results()
            index += 1

        cv2.destroyAllWindows()
        log.info("✅ Image review completed.")
        self._review_flagged_images()
        return self.confirm_save_results()

    def _create_bboxes_on_image(self, img_path, preview_only=False):
        """Displays bounding boxes on an image for visual inspection."""
        metadata_path = self.metadata_dir / img_path.with_suffix(".json").name

        if not metadata_path.exists():
            log.error(f"Metadata file not found: {metadata_path}")
            return False
        if not img_path.exists():
            log.error(f"Image file not found: {img_path}")
            return False

        image = cv2.imread(str(img_path))
        if image is None:
            log.error(f"Error loading image: {img_path}")
            return False

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Set up resizing/scaling parameters
        fullres_h, fullres_w = (9520, 13368)
        downscaling_factor = 5
        resized_h, resized_w = int(image.shape[0] / downscaling_factor), int(image.shape[1] / downscaling_factor)
        scale_x = resized_w / fullres_w
        scale_y = resized_h / fullres_h
        resized_image = cv2.resize(image, (resized_w, resized_h))

        base_lw = max(round(sum(resized_image.shape) / 2 * 0.003), 2)
        label_color = (128, 128, 128)
        text_color = (255, 255, 255)

        for bbox in metadata.get("annotations", []):
            x1, y1, w, h = bbox["bbox_xywh"]
            
            x2, y2 = x1 + w, y1 + h
            x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
            x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 15, 200), 2)

            cat_class_id = str(bbox.get("category_class_id", "Unknown"))
            area_sqm = bbox["global_coordinates"]["area_sqm"]
            area_sqcm = area_sqm * 10000  # convert m² to cm²
            is_primary = bbox.get("is_primary", False)
            
            label_text = f"{self.species_info.get(cat_class_id, 'Unknown')} ({area_sqcm:.2f} cm2) {'P' if is_primary else ''}"

            bbox_height = max(y2 - y1, 1)
            bbox_width = max(x2 - x1, 1)
            font_thickness = max(int(base_lw / 2), 1)

            proposed_scale = bbox_height / 50
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, proposed_scale, font_thickness
            )
            max_text_width = bbox_width * 0.9
            if text_width > max_text_width:
                proposed_scale *= max_text_width / text_width
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, proposed_scale, font_thickness
                )
            font_scale = min(max(proposed_scale, 0.4), 1.5)
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            draw_above = y1 - text_height >= 3

            text_bg_top_left = (x1, y1 - text_height - 3) if draw_above else (x1, y1 + 3)
            text_bg_bottom_right = (x1 + text_width, y1 - 3) if draw_above else (x1 + text_width, y1 + text_height + 6)
            cv2.rectangle(resized_image, text_bg_top_left, text_bg_bottom_right, label_color, -1, cv2.LINE_AA)

            text_org = (x1, y1 - 5) if draw_above else (x1, y1 + text_height + 2)
            cv2.putText(
                resized_image, label_text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                text_color, font_thickness, cv2.LINE_AA
            )
            # 🔹 Add info box with image ID and unique species names
            unique_species_ids = {
                str(bbox.get("category_class_id", "Unknown"))
                for bbox in metadata.get("annotations", [])
            }
            unique_species_names = [
                self.species_info.get(sp_id, "Unknown") for sp_id in unique_species_ids
            ]

            info_lines = [f"Image ID: {img_path.stem}"] + [f"Unique species: {','.join(unique_species_names)}"]
            font_scale = 0.6
            font_thickness = 1
            padding = 10
            line_height = 20

            # Draw background box
            box_width = 0
            for line in info_lines:
                (text_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                box_width = max(box_width, text_width)
            box_height = line_height * len(info_lines)

            cv2.rectangle(
                resized_image,
                (5, 5),
                (5 + box_width + 2 * padding, 5 + box_height + 2 * padding),
                (50, 50, 50),
                thickness=-1
            )

            # Draw text lines
            for i, line in enumerate(info_lines):
                y = 5 + padding + i * line_height + 15
                cv2.putText(
                    resized_image,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA
                )

        if not preview_only:
            cv2.imshow("Inspection Viewer", resized_image)
        
        save_path = self.remapped_sample_dir / img_path.name
        if not save_path.exists():
            cv2.imwrite(str(save_path), resized_image)
        
        return True
    
    def _display_bboxes_on_image(self, img_path: Path):
        """"""
        if img_path.exists():
            image = cv2.imread(str(img_path))
            cv2.imshow("Inspection Viewer", image)
        
        return True

    def generate_all_sample_images(self):
        """Pre-generates and saves annotated sample images before user review."""
        log.info("Generating sample images...")
        count = 0        
        for img_path in tqdm(self.images, desc="Generating sample images"):
            save_path = self.remapped_sample_dir / img_path.name
            if not save_path.exists():
                success = self._create_bboxes_on_image(img_path, preview_only=True)
                if success:
                    count += 1
            else:
                log.debug(f"Sample image already exists: {save_path}")
                continue
        log.info(f"Generated {count} sample images in {self.remapped_sample_dir}")
        
    def _get_user_input(self):
        """Captures user input for labeling images."""
        while True:
            key = cv2.waitKey(0) & 0xFF
            key_char = chr(key)
            if key_char in LABEL_OPTIONS:
                return LABEL_OPTIONS[key_char]
            print("⚠️ Invalid choice. Please press a valid key (1-7, 0, q, or b).")

    def _review_flagged_images(self):
        """Offers the option to review flagged images for issue reporting."""
        df_final = pd.read_csv(self.csv_file)
        flagged_images = df_final[df_final["Selection"] != "Pass"]
        if flagged_images.empty:
            return self.csv_file

        print("\n⚠️ Some images have issues.")
        print(f"📌 Please report them at our GitHub repository: {GITHUB_REPO_URL}")
        if input("Would you like to review the flagged images for screenshots? (y/n): ").strip().lower() == 'y':
            self._display_flagged_images(flagged_images)

        print("\n📌 After taking screenshots, submit an issue on GitHub:")
        print(f"🔗 {GITHUB_REPO_URL}")
        print(f"Title the issue: {self.batch_folder.name} preprocessing inspection: {len(flagged_images)} flagged images\n")
        return self.csv_file

    def _display_flagged_images(self, flagged_images):
        """Displays flagged images for screenshot capture."""
        for _, row in flagged_images.iterrows():
            # Use self.image_dir to locate images based on ImageID
            img_path = self.image_dir / f"{row['ImageID']}.jpg"
            if not img_path.exists():
                img_path = self.image_dir / f"{row['ImageID']}.JPG"
            if img_path.exists():
                image = cv2.imread(str(img_path))
                resized_image = cv2.resize(image, (13376 // 10, 9528 // 10))
                cv2.imshow("Flagged Image", resized_image)
                print(f"📸 Take a screenshot for: {row['ImageID']} ({row['Selection']})")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break
            else:
                print(f"⚠️ Could not find image: {row['ImageID']}")
        cv2.destroyAllWindows()


class PDFReviewer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.use_lts_images = cfg.inspection.use_lts_images

        self.lts_dir = find_lts_dir(cfg.batch_id, cfg.paths.lts_locations, local=False, developed=True, dngs=False, jpgs=True)
        self.lts_batch_dir = Path(self.lts_dir) / "semifield-developed-images" / cfg.batch_id

        self.ms_pdf_report = self.lts_batch_dir / "inspection" / Path(cfg.paths.pdf_report).name if self.use_lts_images else Path(cfg.paths.pdf_report)
        self.inspection_dir = self.lts_batch_dir / "inspection" if self.use_lts_images else Path(cfg.paths.inspection_dir)
        
        self.output_dir = self.inspection_dir / "metashape_report_pages"
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_pdf_images(self, dpi=250):
        """Extracts pages from a PDF and saves them as images."""
        if len(list(self.output_dir.glob("*.jpg"))) > 10:
            log.info("PDF pages already extracted. Skipping extraction.")
            return True
        
        if not self.ms_pdf_report.exists():
            log.error(f"Metashape report not found at: {self.ms_pdf_report}")
            return False
        try:
            doc = fitz.open(self.ms_pdf_report)
            for i in range(len(doc)):
                page = doc.load_page(i)
                zoom = dpi / 72.0  # 72 is the default resolution
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                output_path = self.output_dir / f"page_{i+1:03d}.jpg"
                pix.save(str(output_path))
            log.info(f"Saved PDF pages to: {self.output_dir}")
            doc.close()
            return True
        except Exception as e:
            log.error(f"Failed to extract PDF images: {e}", exc_info=True)
            return False

    def review_pdf(self):
        """Allows a user to review PDF pages as images using a manually set window size,
        with an option to go back to the previous page."""
        # Set desired window dimensions (adjust these values as needed)
        manual_width, manual_height = 1000, 1200

        window_name = "PDF Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, manual_width, manual_height)
        
        pdf_images = sorted(self.output_dir.glob("*.jpg"))
        if not pdf_images:
            log.error("No PDF pages found for review.")
            return

        index = 0
        log.info(f"Reviewing PDF pages.")

        print("🖼️  Press any key to go to next page, 'b' to go back, 'q' to quit.")
        while index < len(pdf_images):
            image = cv2.imread(str(pdf_images[index]))
            if image is None:
                index += 1
                continue

            # Scale the image to fit within the manually set window dimensions.
            h, w = image.shape[:2]
            scale_factor = min(manual_width / w, manual_height / h, 1.0)
            if scale_factor < 1.0:
                image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))

            cv2.imshow(window_name, image)
            
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('b'):
                if index > 0:
                    index -= 1
                else:
                    print("Already at the first page; cannot go back further.")
            else:
                index += 1

        cv2.destroyAllWindows()
        log.info("PDF review completed.")

class ReviewSession:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.image_reviewer = ImageReviewer(cfg)
        self.pdf_reviewer = PDFReviewer(cfg)

    def run(self):
        """Prompt the user to select a review mode and run the corresponding review."""
        log.info("Starting interactive review session.")
        while True:
            print("\nSelect review mode:")
            print("1 - Image Review")
            print("2 - PDF Review")
            print("3 - Both")
            print("q - Quit")
            choice = input("Your choice: ").strip().lower()
            log.info(f"Review mode selected: {choice}")
            if choice == "1":
                self.image_reviewer.review_images()
            elif choice == "2":
                self.pdf_reviewer.review_pdf()
            elif choice == "3":
                self.image_reviewer.review_images()
                self.pdf_reviewer.review_pdf()
            elif choice == "q":
                print("Exiting review session.")
                break
            else:
                print("Invalid choice. Please select again.")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    log.info("Creating inspection images...")
    try:
        log.info("Starting image review...")
        image_reviewer = ImageReviewer(cfg)
        image_reviewer.generate_all_sample_images()
        log.info("Sample images generated.")
        
        log.info("Starting image review...")
        pdf_reviewer = PDFReviewer(cfg)
        pdf_reviewer.extract_pdf_images()
        log.info("PDF pages extracted.")

        log.info("Starting annotation summary plots...")
        plotter = AnnotationPlotter(cfg)
        plotter.plot_summary()
        log.info("Summary plots generated.")

        if cfg.inspection.inspect:
            log.info("Starting review session...")
            session = ReviewSession(cfg)
            session.run()
            log.info("Review session completed.")
    
    except Exception as e:
        log.exception(f"Inspection failed: {e}")

if __name__ == "__main__":
    main()
