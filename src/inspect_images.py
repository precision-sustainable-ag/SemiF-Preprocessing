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

def show_images(folder_path, timestamp=None, user=None):
    folder = Path(folder_path)
    images = sorted(list(folder.glob('*.jpg')) + list(folder.glob('*.JPG')))

    if not images:
        log.warning(f"No images found in {folder_path}")
        return
    
    csv_file = f'{folder_path.parent}/preprocessing_inspection_results.csv'
    
    # Load existing labeled images if CSV exists
    if Path(csv_file).exists():
        df_existing = pd.read_csv(csv_file)
        labeled_images = set(df_existing['Image Path'].tolist())  # Track already labeled images
        results = df_existing.values.tolist()  # Continue appending to existing records
    else:
        labeled_images = set()
        results = []

    unlabeled_images = [img for img in images if img.stem not in labeled_images]

    if not unlabeled_images:
        log.info("All images have been labeled. Exiting.")
        return

    # print all the options for the user in the terminal
    print("Press 'a' to pass")
    print("Press 's' to fail")
    print("Press 'd' to flag")
    print("Press 'q' to quit")
    
    index = 0

    while index < len(unlabeled_images):
        img_path = unlabeled_images[index]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error loading image: {img_path}")
            index += 1
            continue
        
        # Resize image to fit screen
        resized_image = cv2.resize(image, (800, 600))
        
        cv2.imshow("Image Viewer", resized_image)
        key = cv2.waitKey(0) & 0xFF  # Wait for key press

        if key in [ord('a'), ord('s'), ord('d')]:  # Move to next image on valid key press
            if key == ord('a'):
                selection = "pass"
            elif key == ord('s'):
                selection = "fail"
            elif key == ord('d'):
                selection = "flagged"
            
            results.append([img_path.stem, selection, timestamp, user])
            df = pd.DataFrame(results, columns=['Image Path', 'Selection', 'Timestamp', 'User'])
            df = df.sort_values(by='Image Path').reset_index(drop=True)
            df.to_csv(csv_file, index=False)
            log.debug(f"Saved: {img_path} - {selection}")
            
            index += 1  # Move to the next image
        elif key == ord('q'):  # Quit
            break
        else:
            log.warning(f"Invalid key: {key}")

    cv2.destroyAllWindows()
    print("Image review completed.")

    # Check for any failed or review images
    df_final = pd.read_csv(csv_file)
    failed_or_review = df_final[df_final["Selection"].isin(["fail", "flagged"])]

    if not failed_or_review.empty:
        log.warning(f"Failed/Reviewed images: {failed_or_review['Image Path'].tolist()}")
        print("\n⚠️ Some images were marked as 'fail' or 'flagged'.")
        print(f"📌 Please create an issue in our GitHub repository: {GITHUB_REPO_URL}")
        print("Mention the failed/flagged images and describe any issues you encountered.")
        print(f"Title the issue: 'Failed/Flagged images for batch {folder_path.parent.name}'\n")
        # Offer to display failed/review images for screenshot capture
        show_flagged = input("Would you like to review the failed/review images for screenshots? (y/n): ").strip().lower()

        if show_flagged == 'y':
            for _, row in failed_or_review.iterrows():
                img_path = folder / f"{row['Image Path']}.jpg"  # Assuming .jpg, modify if needed
                if not Path(img_path).exists():
                    img_path = folder / f"{row['Image Path']}.JPG"  # Try uppercase

                if Path(img_path).exists():
                    image = cv2.imread(str(img_path))
                    resized_image = cv2.resize(image, (800, 600))
                    
                    cv2.imshow("Flagged Image", resized_image)
                    print(f"Displaying: {row['Image Path']} - {row['Selection']}")
                    print("📸 Take a screenshot for documentation.")
                    print("Press any key to move to the next image.")
                    
                    key = cv2.waitKey(0) & 0xFF  # Wait for key press to move to next
                    if key == ord('q'):  # Allow early exit
                        break
                else:
                    print(f"⚠️ Could not find image: {row['Image Path']}")

            cv2.destroyAllWindows()
        
        print("\n📌 Once you've taken screenshots, submit an issue on GitHub:")
        print(f"🔗 {GITHUB_REPO_URL}\n")

    return csv_file


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for RAW to JPG conversion."""
    log.info("Inspecting images...")
    batch_id = cfg.batch_id
    lts_locations = cfg.paths.lts_locations
    
    for lts_location in lts_locations:
        nfs_location = Path(lts_location)
        batch_location = Path(cfg.paths.data_dir) / nfs_location.name / "semifield-developed-images" / batch_id
        if batch_location.exists():
            log.info(f"Batch {batch_id} found in {batch_location}")
            break
    
    # Local JPG sample directory
    local_sample_dir = batch_location / "sample_images"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user = getpass.getuser()
    src_csv_file = show_images(local_sample_dir, timestamp, user)
    lts_location = [x for x in cfg.paths.lts_locations if nfs_location.name == Path(x).name][0]
    dst_lts_batch_location = Path(lts_location) / "semifield-developed-images" / batch_id

    # Copy CSV file to LTS location
    dst_csv_file = dst_lts_batch_location / Path(src_csv_file).name
    shutil.copy(src_csv_file, dst_csv_file)




    
if __name__ == "__main__":
    main()
