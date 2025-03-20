import cv2
import pandas as pd
import datetime
import getpass
from pathlib import Path
import hydra
from omegaconf import DictConfig
from src.utils.utils import find_lts_dir
import logging

log = logging.getLogger(__name__)

GITHUB_REPO_URL = "https://github.com/precision-sustainable-ag/SemiF-Preprocessing/issues"

LABEL_OPTIONS = {
    "1": "Pass",
    "2": "Preprocessing Quality",
    "3": "Potting Area Cleanliness",
    "4": "Non-Target",
    "5": "Plant Spacing",
    "0": "Other",
    "q": "Quit",
    "b": "Back"
}

class ImageReviewer:
    def __init__(self, cfg: DictConfig):
        self.batch_id = cfg.batch_id
        self.lts_locations = cfg.paths.lts_locations
        self.lts_dir = find_lts_dir(self.batch_id, self.lts_locations, local=False, developed=True, dngs=False, jpgs=True)
        self.lts_dir_name = Path(self.lts_dir).name
        self.batch_folder = Path(self.lts_dir) / "semifield-developed-images" / self.batch_id
        self.lts_sample_dir = self.batch_folder / "preprocessing_samples"
        self.csv_file = self.batch_folder / f"{self.batch_id}_preprocessing_inspection_results.csv"
        self.images = self._load_images()
        self.results = self._load_existing_results()
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.user = getpass.getuser()

    def _load_images(self):
        """Load images and return a sorted list of unlabeled ones."""
        all_images = sorted(self.lts_sample_dir.glob("*.jpg")) + sorted(self.lts_sample_dir.glob("*.JPG"))
        if not all_images:
            log.warning(f"No images found in {self.lts_sample_dir}")
            return []

        return [img for img in all_images if img.stem not in self._get_labeled_images()]

    def _get_labeled_images(self):
        """Retrieve a set of already labeled images from the CSV file."""
        if self.csv_file.exists():
            df_existing = pd.read_csv(self.csv_file)
            return set(df_existing['ImageID'].tolist())
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
            if key == "0":
                key = "0 (zero)"
            bright_key = f"\033[1;97m{key}\033[0m"  # Makes numbers bold & bright white
            
            print(f"{bright_key} - {label}")
        
        print("\n🔄 Please wait while the X11 or X410 forwarding initializes. This may take a few seconds...\n")

    def _confirm_save_results(self):
        """Ask the user if they want to save the final CSV. If not, delete the file."""
        while True:
            confirm = input("\n💾 Do you want to save the final inspection results? (y/n): ").strip().lower()
            
            if confirm == "y":
                self._save_results()
                log.info(f"✅ Inspection results saved to {self.csv_file}")
                return self.csv_file  # File saved successfully

            elif confirm == "n":
                if self.csv_file.exists():
                    if self.csv_file.name == f"{self.batch_id}_preprocessing_inspection_results.csv":
                        self.csv_file.unlink()  # Delete the CSV
                        log.info(f"❌ Inspection results discarded. {self.csv_file} removed.")
                else:
                    log.warning("⚠️ No saved CSV file found to delete.")
                return None  # User discarded results

            else:
                print("⚠️ Invalid input. Please enter 'y' to save or 'n' to discard.")
                
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
            if label == "Quit":
                print("\n❌ Exiting image review.")
                cv2.destroyAllWindows()
                return self.csv_file  # Save progress and exit
            
            if label == "Back":
                if index > 0:
                    print("\n🔙 Going back to the previous image.")
                    self.results.pop()  # Remove last entry
                    index -= 1  # Move back an index
                else:
                    print("⚠️ Already at the first image, cannot go back further.")
                continue  # Restart loop without saving

            self.results.append([self.batch_id, img_path.stem, label, self.timestamp, self.user, self.lts_dir_name])
            self._save_results()
            index += 1

        cv2.destroyAllWindows()
        log.info("✅ Image review completed.")
        self._review_flagged_images()
        return self._confirm_save_results()

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
        df = pd.DataFrame(self.results, columns=['BatchID', 'ImageID', 'Selection', 'Timestamp', 'User', 'LTSLocation'])
        df.to_csv(self.csv_file, index=False)

    def _review_flagged_images(self):
        """Checks and offers to display flagged images for issue reporting."""
        df_final = pd.read_csv(self.csv_file)
        flagged_images = df_final[df_final["Selection"] != "Pass"]

        if flagged_images.empty:
            return self.csv_file

        print("\n⚠️ Some images have issues.")
        print(f"📌 Please report in our GitHub repository: {GITHUB_REPO_URL}")
        print("Mention the flagged images and describe the issues.")
        
        if input("Would you like to review the flagged images for screenshots? (y/n): ").strip().lower() == 'y':
            self._display_flagged_images(flagged_images)

        print("\n📌 After taking screenshots, submit an issue on GitHub:")
        print(f"🔗 {GITHUB_REPO_URL}\n")
        print(f"Title the issue: {self.batch_folder.name} preprocessing inspection: {len(flagged_images)} flagged images\n")
        return self.csv_file

    def _display_flagged_images(self, flagged_images):
        """Displays flagged images for screenshot capture."""
        for _, row in flagged_images.iterrows():
            img_path = self.lts_sample_dir / f"{row['ImageID']}.jpg"
            if not img_path.exists():
                img_path = self.lts_sample_dir / f"{row['ImageID']}.JPG"

            if img_path.exists():
                image = cv2.imread(str(img_path))
                resized_image = cv2.resize(image, (13376 // 10, 9528 // 10))
                cv2.imshow("Flagged Image", resized_image)
                print(f"📸 Take a screenshot for: {row['ImageID']} ({row['Selection']})")

                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):  # Allow early exit
                    break
            else:
                print(f"⚠️ Could not find image: {row['ImageID']}")

        cv2.destroyAllWindows()


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for image quality inspection."""
    log.info("🔍 Inspecting images...")

    reviewer = ImageReviewer(cfg)
    src_csv_file = reviewer.review_images()
    log.info(f"Inspection results saved to {src_csv_file}")
    log.info("Image inspection completed.")

if __name__ == "__main__":
    main()
