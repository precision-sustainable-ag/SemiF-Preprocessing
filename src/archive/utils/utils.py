import os
from pathlib import Path
from typing import List


def locate_lts_location(lts_locations: List[str], batch_id: str, type: str) -> (
        Path):
    """
    Util function to locate batch uploads
    Args:
        lts_locations (list): nfs storage locations
        batch_id (str): batch id
        type (str): semifield-upload / semifield-developed-images
    Returns:
        batch_folder (Path): batch lts location for specified type
    """

    batch_folder = None
    for path in lts_locations:
        semif_uploads = os.path.join(path, type)
        batch_ids = [x.name for x in Path(semif_uploads).glob("*")]
        if batch_id in batch_ids:
            batch_folder = Path(semif_uploads) / batch_id
            break
    return batch_folder

def create_developed_images(uploads_folder: Path) -> Path:
    """
    Works under the assumption that developed images don't exist
    Creates and returns a folder for developed images at the same lts location
    Args:
        uploads_folder (Path): located uploads folder for the batch
    Returns:
        developed_images_path (Path): newly created folder for developed images
    """
    lts_location = Path(uploads_folder).parent.parent
    batch_id = uploads_folder.name
    developed_images_folder = lts_location / "semifield-developed-images" / batch_id
    os.makedirs(developed_images_folder, exist_ok=True)
    return developed_images_folder