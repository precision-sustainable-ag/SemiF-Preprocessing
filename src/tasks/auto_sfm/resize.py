import glob
import logging
import os
from math import ceil
from multiprocessing import Pool
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import piexif
from PIL import Image, ImageFile
from skimage.color import rgb2hsv
from skimage.morphology import binary_closing, square

from src.utils.utils import retry_nfs_access, find_lts_dir

ImageFile.LOAD_TRUNCATED_IMAGES = True

log = logging.getLogger(__name__)


def remove_missing_data(cfg):
    """Removes missing downscaled images are masks.

    Args:
        cfg (DictOmega): Hydra object
    """

    imgs = glob.glob(os.path.join(cfg.asfm.down_photos, "*.jpg"))
    masks = glob.glob(os.path.join(cfg.asfm.down_masks, "*.png"))

    imgs = [Path(img).stem for img in imgs]
    masks = [Path(mask).stem.replace("_mask", "") for mask in masks]

    # find the missing and additional elements in masks
    miss = list(set(imgs).difference(masks))
    add = list(set(masks).difference(imgs))

    if (len(miss) == 0) and (len(add) == 0):
        None
    elif len(miss) > len(add):
        # More photos than masks. Remove extra photos
        log.warning(
            f"More photos than masks. Removing {len(miss)} extra down_scaled photos"
        )
        for img in miss:
            Path(cfg.asfm.down_photos, img + ".jpg").unlink()
    elif len(add) > len(miss):
        # More masks than photos. Remove extra masks
        log.warning(
            f"More masks than photos. Removing {len(add)} extra down_scaled masks"
        )
        for mask in add:
            Path(cfg.asfm.down_masks, mask + "_mask.png").unlink()

def fix_exif_types(exif_dict):
    for ifd in ("0th", "Exif", "GPS", "1st"):
        if ifd in exif_dict:
            fixed = {}
            for tag, value in exif_dict[ifd].items():

                # Fix SRational[] for tag 50721
                if tag == 50721 and isinstance(value, tuple) and all(isinstance(v, tuple) and len(v) == 2 for v in value):
                    fixed[tag] = list(value)
                    log.debug(f"Fixed tag {tag} from {value} to {fixed[tag]}")
                    continue

                # Fix Short[] for tag 50728
                if tag == 50728:
                    if isinstance(value, tuple) and all(
                        isinstance(v, tuple) and len(v) == 2 and v[1] != 0 for v in value
                    ):
                        # Convert SRationals to ints (numerator // denominator)
                        fixed[tag] = [int(v[0] / v[1]) for v in value]
                        log.debug(f"Fixed tag {tag} from {value} to {fixed[tag]}")
                        continue

                # Fix BlackLevel (SRational single value)
                if tag == 50714 and isinstance(value, int):
                    fixed[tag] = (value, 1)
                    log.debug(f"Fixed tag {tag} from {value} to {fixed[tag]}")
                    continue

                # General valid types
                if isinstance(value, (int, str, bytes)):
                    fixed[tag] = value
                elif isinstance(value, tuple) and all(isinstance(v, int) for v in value):
                    fixed[tag] = value
                elif isinstance(value, list) and all(isinstance(v, int) for v in value):
                    fixed[tag] = value
                else:
                    log.debug(f"Skipping tag {tag} due to bad type: {type(value)} -> {value}")
                    continue

            exif_dict[ifd] = fixed
    return exif_dict

def resize_image(image_src, scale, masks):
    """Resize an image based on the given scale."""
    assert 0.0 < scale <= 1.0, "scale should be between (0, 1]."

    try:
        with Image.open(image_src) as image:
            width, height = image.size
            scaled_width = int(ceil(width * scale))
            scaled_height = int(ceil(height * scale))
            resized_image = image.resize((scaled_width, scaled_height))
            return resized_image, image.copy()
    except (IOError, SyntaxError) as e:
        log.error(f"Bad file: {image_src}. Error: {e}")
        return None, None


def save_resized_image(resized_image, image, image_dst, masks, bbot_version):
    """Save the resized image to the destination path."""
    kwargs = {}
    try:
        if masks:
            resized_image.save(image_dst, quality=95, **kwargs)
        else:
            try:
                exif_data = piexif.load(image.info["exif"])
                # if "3.1" in bbot_version:
                exif_data = fix_exif_types(exif_data)
                exif_data = piexif.dump(exif_data)
                kwargs["exif"] = exif_data
            except KeyError:
                log.warning("EXIF data not found, resizing without EXIF data.")

            resized_image.save(image_dst, quality=95, **kwargs)
    except Exception as e:
        log.error(f"Error saving file: {image_dst}. Error: {e}")

def resize_and_save(data):
    """Resize and save an image with a single retry via retry_nfs_access on PermissionError."""
    image_src = Path(data["image_src"])
    image_dst = Path(data["image_dst"])
    scale = data["scale"]
    masks = data["masks"]
    bbot_version = data["bbot_version"]

    try:
        resized_image, image = resize_image(image_src, scale, masks)
    except PermissionError as e:
        log.warning(f"Permission denied on {image_src}. Attempting NFS retry.")
        success = retry_nfs_access(image_src, mode="read")
        if not success:
            log.error(f"NFS access failed after retries for {image_src}. Skipping.")
            return
        try:
            resized_image, image = resize_image(image_src, scale, masks)
        except Exception as e:
            log.error(f"Second attempt failed for {image_src}. Error: {e}")
            return
    except Exception as e:
        log.error(f"Error resizing image {image_src}. Error: {e}")
        return

    if resized_image and image:
        save_resized_image(resized_image, image, image_dst, masks, bbot_version)

def resize_photo_diretory(cfg):
    # base_path = Path(cfg.paths.images)
    lts_dir = find_lts_dir(cfg.batch_id, cfg.paths.lts_locations, developed=True, jpgs=True)
    base_path = Path(lts_dir) / "semifield-developed-images" / cfg.batch_id / "images"

    save_dir = Path(cfg.paths.down_photos)
        

    files = sorted(list(base_path.glob("*.jpg")) + list(base_path.glob("*.JPG")))
    num_files = len(files)

    if save_dir.exists():
        already_copied_images = list(save_dir.glob("*.jpg")) + list(save_dir.glob("*.JPG"))
        num_already_copied_images = len(already_copied_images)
        if num_already_copied_images == num_files:
            log.debug(f"All images ({num_already_copied_images}) have already been resized.")
            return
        elif num_already_copied_images < num_files:
            log.debug(f"{num_already_copied_images} images have already been resized, {num_files - num_already_copied_images} images remaining.")
            files = [file for file in files if file.name not in [img.name for img in already_copied_images]] 

    data = [
        {
            "image_src": src,
            "image_dst": save_dir / src.name,
            "scale": cfg.asfm.downscale.factor,
            "masks": False,
            "bbot_version": cfg.bbot_version,
        }
        for src in files
    ]

    try:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(resize_and_save, item): item for item in data}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                    print(f"Progress: {i}/{num_files} images resized")
                except Exception as e:
                    log.error(f"An error occurred while resizing: {e}")
    except KeyboardInterrupt:
        log.info("Interrupted by user, terminating...")
    except Exception as e:
        log.error(f"An error occurred: {e}")
    finally:
        log.info("Completed resizing images.")


def create_masks(cfg):
    down_photos_dir = cfg["asfm"]["down_photos"]
    save_dir = cfg["asfm"]["down_masks"]
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    img_files = glob.glob(os.path.join(down_photos_dir, "*.jpg"))
    num_files = len(img_files)
    log.info(f"Masking {num_files} images.")
    data = [
        {
            "image_src": src,
            "mask_dst": str(Path(save_dir, Path(src).stem + "_mask.png")),
        }
        for src in img_files
    ]

    # Adjust the number of processes as needed
    num_processes = int(len(os.sched_getaffinity(0)) / cfg.max_workers)

    # try:
    with Pool(num_processes) as pool:
        for i, _ in enumerate(pool.imap_unordered(mask_img, data), 1):
            # pool.imap_unordered(resize_and_save, data)
            print(f"Progress: {i}/{num_files} images masked.")
    # except KeyboardInterrupt:
    #     log.info("Interrupted by user, terminating...")
    #     pool.terminate()
    # except Exception as e:
    #     log.error(f"An error occurred: {e}")
    # finally:
    #     log.info("Completed masking images.")


def simple_mask(mask):
    # print(closed_mask)
    # Calculate the number of pixels for 5% of the height of the image
    percent = 0.15
    height_percent = int(mask.shape[0] * percent)
    # Set the top 5% and bottom 5% of the mask to false (unmasked)
    if mask.max() == 255:
        if mask[:height_percent, :].max() == 255:
            mask[:height_percent] = 255

        if mask[-height_percent:, :].max() == 255:
            mask[-height_percent:, :] = 255
    return mask


def mask_img(data):
    image_src = data["image_src"]
    mask_dst = data["mask_dst"]

    image = cv2.cvtColor(cv2.imread(image_src), cv2.COLOR_BGR2RGB)
    # Convert the image to HSV
    hsv_image = rgb2hsv(image)
    # Define the range for blue color
    # These ranges can be adjusted depending on the shade of blue in the image
    lower_blue = np.array([0.4, 0.3, 0.2])
    upper_blue = np.array([0.6, 0.9, 1])

    # Create a binary mask for the blue color
    mask = (
        (hsv_image[:, :, 0] >= lower_blue[0])
        & (hsv_image[:, :, 0] <= upper_blue[0])
        & (hsv_image[:, :, 1] >= lower_blue[1])
        & (hsv_image[:, :, 1] <= upper_blue[1])
        & (hsv_image[:, :, 2] >= lower_blue[2])
        & (hsv_image[:, :, 2] <= upper_blue[2])
    )

    # Convert the mask to uint8 format
    mask = mask.astype(np.uint8)

    kernel = square(35)
    closed_mask = binary_closing(mask, kernel).astype(np.uint8)
    closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.dilate(closed_mask, kernel, iterations=3) * 255
    masked = simple_mask(closed_mask)

    cv2.imwrite(mask_dst, masked)

    return True


def resize_masks(cfg):
    base_path = cfg["batchdata"]["masks"]
    save_dir = cfg["asfm"]["down_masks"]

    files = glob.glob(os.path.join(base_path, "*.png"))
    num_files = len(files)
    log.debug(f"Found {num_files} files to process.")

    data = [
        {
            "image_src": src,
            "image_dst": os.path.join(save_dir, os.path.basename(src)),
            "scale": cfg["asfm"]["downscale"]["factor"],
            "masks": True,
        }
        for src in files
    ]

    # Adjust the number of processes as needed
    num_processes = int(len(os.sched_getaffinity(0)) / cfg.general.cpu_denominator)

    try:
        with Pool(num_processes) as pool:
            # pool.imap_unordered(resize_and_save, data)
            for i, _ in enumerate(pool.imap_unordered(resize_and_save, data), 1):
                print(f"Progress: {i}/{num_files} masks resized")
    except KeyboardInterrupt:
        log.info("Interrupted by user, terminating...")
        pool.terminate()
    except Exception as e:
        log.error(f"An error occurred: {e}")
    finally:
        log.info("Completed resizing masks.")

    remove_missing_data(cfg)
