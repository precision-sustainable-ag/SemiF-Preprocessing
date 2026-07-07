import glob
import logging
import os
from math import ceil
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image, ImageFile
from skimage.color import rgb2hsv
from skimage.morphology import binary_closing, square

from src.utils.utils import retry_nfs_access, find_lts_dir, get_files

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 200_000_000

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

from math import ceil
from pathlib import Path

from PIL import Image


def resize_image(image_src, scale):
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


def build_save_kwargs_from_source_image(image):
    """
    Build safe Pillow save kwargs.

    For non-mask images, preserve the original raw EXIF blob and ICC profile.
    For masks, do not preserve EXIF/ICC metadata.
    """
    kwargs = {
        "quality": 100,
    }

    if "exif" in image.info:
        kwargs["exif"] = image.info["exif"]
    else:
        log.warning("EXIF data not found, resizing without EXIF data.")

    if "icc_profile" in image.info:
        kwargs["icc_profile"] = image.info["icc_profile"]
    else:
        log.warning("ICC profile not found, resizing without ICC profile.")

    return kwargs


def update_exif_dimension_tags(image_dst, width, height):
    """
    Update EXIF dimension tags after saving a resized image.

    Requires exiftool on PATH.
    """
    if shutil.which("exiftool") is None:
        log.warning(
            "exiftool not found; skipping EXIF dimension tag updates for %s",
            image_dst,
        )
        return

    try:
        subprocess.run(
            [
                "exiftool",
                "-overwrite_original",
                f"-ExifImageWidth={width}",
                f"-ExifImageHeight={height}",
                f"-PixelXDimension={width}",
                f"-PixelYDimension={height}",
                str(image_dst),
            ],
            check=True,
            text=True,
            capture_output=True,
        )

    except subprocess.CalledProcessError as e:
        log.error(
            "Failed to update EXIF dimension tags for %s. Error: %s",
            image_dst,
            e.stderr,
        )


def save_resized_image(
    resized_image,
    image,
    image_dst,
    bbot_version=None,
    update_dimensions=True,
):
    """
    Save the resized image to the destination path.

    Uses direct EXIF/ICC metadata copy instead of piexif load/dump.
    Optionally updates EXIF dimension tags after save.
    """
    try:
        kwargs = build_save_kwargs_from_source_image(
            image=image,
        )

        resized_image.save(image_dst, **kwargs)

        if update_dimensions:
            width, height = resized_image.size
            update_exif_dimension_tags(
                image_dst=image_dst,
                width=width,
                height=height,
            )

    except Exception as e:
        log.error(f"Error saving file: {image_dst}. Error: {e}")

def resize_and_save(data):
    """Resize and save an image with a single retry via retry_nfs_access on PermissionError."""
    image_src = Path(data["image_src"])
    image_dst = Path(data["image_dst"])
    scale = data["scale"]
    bbot_version = data["bbot_version"]

    try:
        resized_image, image = resize_image(image_src, scale)
    except PermissionError as e:
        log.warning(f"Permission denied on {image_src}. Attempting NFS retry.")
        success = retry_nfs_access(image_src, mode="read")
        if not success:
            log.error(f"NFS access failed after retries for {image_src}. Skipping.")
            return
        try:
            resized_image, image = resize_image(image_src, scale)
        except Exception as e:
            log.error(f"Second attempt failed for {image_src}. Error: {e}")
            return
    except Exception as e:
        log.error(f"Error resizing image {image_src}. Error: {e}")
        return

    if resized_image and image:
        save_resized_image(resized_image, image, image_dst, bbot_version=bbot_version)

def resize_photo_diretory(cfg):
    # base_path = Path(cfg.paths.images)
    lts_dir = find_lts_dir(cfg.batch_id, cfg.paths.lts_locations, developed=True, jpgs=True)
    base_path = Path(lts_dir) / "semifield-developed-images" / cfg.batch_id / "images"

    save_dir = Path(cfg.paths.down_photos)
        

    files = get_files(cfg, task="auto_sfm_resize")
    num_files = len(files)

    overwrite = getattr(cfg.asfm.downscale, "overwrite", False)

    if save_dir.exists() and not overwrite:
        already_copied_images = list(save_dir.glob("*.jpg")) + list(save_dir.glob("*.JPG"))
        already_copied_names = {img.name for img in already_copied_images}

        num_already_copied_images = len(already_copied_images)

        if num_already_copied_images == num_files:
            log.debug(f"All images ({num_already_copied_images}) have already been resized.")
            return

        if num_already_copied_images < num_files:
            log.debug(
                f"{num_already_copied_images} images have already been resized, "
                f"{num_files - num_already_copied_images} images remaining."
            )
            files = [file for file in files if file.name not in already_copied_names]
    else:
        log.warning("Overwrite enabled; existing resized images will be regenerated.")

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

