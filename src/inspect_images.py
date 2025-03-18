import cv2
import pandas as pd
import datetime
import getpass
from pathlib import Path
import hydra
import shutil
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

GITHUB_REPO_URL = "https://github.com/precision-sustainable-ag/SemiF-Preprocessing/issues"

LABEL_OPTIONS = {
    "1": "Pass ✅",
    "2": "Preprocessing Quality 🎨",
    "3": "Potting Area Cleanliness 🧹",
    "4": "Non-Target 🌿",
    "5": "Plant Spacing 🌱",
    "0": "Other 📝",
    "q": "Quit ❌"
}

class ImageReviewer:
    def __init__(self, batch_folder):
        self.batch_folder = Path(batch_folder)
        self.local_sample_dir = self.batch_folder / "sample_images"
        self.csv_file = self.batch_folder / "preprocessing_inspection_results.csv"
        self.images = self._load_images()
        self.results = self._load_existing_results()
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.user = getpass.getuser()

    def _load_images(self):
        """Load images and return a sorted list of unlabeled ones."""
        all_images = sorted(self.local_sample_dir.glob("*.jpg")) + sorted(self.local_sample_dir.glob("*.JPG"))
        if not all_images:
            log.warning(f"No images found in {self.local_sample_dir}")
            return []

        return [img for img in all_images if img.stem not in self._get_labeled_images()]

    def _get_labeled_images(self):
        """Retrieve a set of already labeled images from the CSV file."""
        if self.csv_file.exists():
            df_existing = pd.read_csv(self.csv_file)
            return set(df_existing['Image Path'].tolist())
        return set()

    def _load_existing_results(self):
        """Load existing CSV results or return an empty list."""
        if self.csv_file.exists():
            log.info(f"📄 Loading existing results from {self.csv_file}")
            return pd.read_csv(self.csv_file).values.tolist()
        return []

    def display_instructions(self):
        """Prints instructions for user input."""
        print("\n--- Image Quality Assessment ---")
        for key, label in LABEL_OPTIONS.items():
            print(f"{key}️ - {label}")
        print("\n🔄 Please wait while the X11 or X410 forwarding initializes. This may take a few seconds...\n")

    def review_images(self):
        """Iterate over images and allow the user to label them."""
        if not self.images:
            log.info("✅ All images have been labeled. Exiting.")
            return None

        self.display_instructions()
        cv2.namedWindow("Inspection Viewer")

        index = 0
        while index < len(self.images):
            img_path = self.images[index]
            if not self._display_image(img_path):
                index += 1
                continue

            label = self._get_user_input()
            if label == "Quit ❌":
                print("\n❌ Exiting image review.")
                cv2.destroyAllWindows()
                return self.csv_file  # Save progress and exit

            self.results.append([img_path.stem, label, self.timestamp, self.user])
            self._save_results()
            index += 1

        cv2.destroyAllWindows()
        log.info("✅ Image review completed.")
        return self._review_flagged_images()

    def _display_image(self, img_path):
        """Loads and displays an image, returns False if loading fails."""
        image = cv2.imread(str(img_path))
        if image is None:
            log.error(f"⚠️ Error loading image: {img_path}")
            return False

        resized_image = cv2.resize(image, (13376 // 10, 9528 // 10))
        cv2.imshow("Inspection Viewer", resized_image)
        return True

    def _get_user_input(self):
        """Captures user input for labeling images."""
        while True:
            key = cv2.waitKey(0) & 0xFF
            key_char = chr(key)
            if key_char in LABEL_OPTIONS:
                return LABEL_OPTIONS[key_char]
            print("⚠️ Invalid choice. Please press a number between 1-5 or 'q' to quit.")

    def _save_results(self):
        """Save the labeling results to a CSV file."""
        df = pd.DataFrame(self.results, columns=['Image Path', 'Selection', 'Timestamp', 'User'])
        df.to_csv(self.csv_file, index=False)

    def _review_flagged_images(self):
        """Checks and offers to display flagged images for issue reporting."""
        df_final = pd.read_csv(self.csv_file)
        flagged_images = df_final[df_final["Selection"] != "Pass ✅"]

        if flagged_images.empty:
            return self.csv_file

        print("\n⚠️ Some images have issues.")
        print(f"📌 Please report in our GitHub repository: {GITHUB_REPO_URL}")
        print("Mention the flagged images and describe the issues.")
        
        if input("Would you like to review the flagged images for screenshots? (y/n): ").strip().lower() == 'y':
            self._display_flagged_images(flagged_images)

        print("\n📌 After taking screenshots, submit an issue on GitHub:")
        print(f"🔗 {GITHUB_REPO_URL}\n")
        print(f"Title the issue: {self.batch_folder.name} {len(flagged_images)} flagged images\n")
        return self.csv_file

    def _display_flagged_images(self, flagged_images):
        """Displays flagged images for screenshot capture."""
        for _, row in flagged_images.iterrows():
            img_path = self.local_sample_dir / f"{row['Image Path']}.jpg"
            if not img_path.exists():
                img_path = self.local_sample_dir / f"{row['Image Path']}.JPG"

            if img_path.exists():
                image = cv2.imread(str(img_path))
                resized_image = cv2.resize(image, (13376 // 10, 9528 // 10))
                cv2.imshow("Flagged Image", resized_image)
                print(f"📸 Take a screenshot for: {row['Image Path']} ({row['Selection']})")

                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):  # Allow early exit
                    break
            else:
                print(f"⚠️ Could not find image: {row['Image Path']}")

        cv2.destroyAllWindows()


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for image quality inspection."""
    log.info("🔍 Inspecting images...")

    batch_id = cfg.batch_id
    lts_locations = cfg.paths.lts_locations

    for lts_location in lts_locations:
        nfs_location = Path(lts_location)
        batch_location = Path(cfg.paths.data_dir) / nfs_location.name / "semifield-developed-images" / batch_id
        if batch_location.exists():
            log.info(f"✅ Batch {batch_id} found in {batch_location}")
            break

    reviewer = ImageReviewer(batch_location)
    src_csv_file = reviewer.review_images()
    if not src_csv_file:
        return

    lts_location = next((x for x in cfg.paths.lts_locations if nfs_location.name == Path(x).name), None)
    if not lts_location:
        log.error("⚠️ Could not determine LTS location.")
        return

    dst_lts_batch_location = Path(lts_location) / "semifield-developed-images" / batch_id
    dst_csv_file = dst_lts_batch_location / Path(src_csv_file).name

    shutil.copy(src_csv_file, dst_csv_file)
    log.info(f"✅ CSV file copied to LTS: {dst_csv_file}")

if __name__ == "__main__":
    main()
