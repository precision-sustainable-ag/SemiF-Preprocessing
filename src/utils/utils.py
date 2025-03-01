from pathlib import Path
import logging
import numpy as np
log = logging.getLogger(__name__)

# Find the batch NFS location from a list of possible parent directories
def find_lts_dir(batch_id, nfs_locations, local=False, developed=False):
    """
    Searches for the specified batch directory within the given NFS locations and checks for the presence and completeness of RAW files.
    Args:
        batch_id (str): The identifier of the batch to search for.
        nfs_locations (list): A list of NFS locations (directories) to search within.
    Returns:
        Path: The NFS location where the batch was found with complete RAW
        files, or None if the batch is not found or the files are incomplete.
    Logs:
        - Info: Logs the NFS location and the number of RAW files found if the batch is found and the files are complete.
        - Error: Logs an error message if the batch is not found, if no RAW files are found, or if the RAW files are not completely uploaded.
    """
    dir_found = False
    files_found = False
    upload_complete = False
    for nfs_location in nfs_locations:
        nfs_location = Path(nfs_location)
        if local:
            if developed:
                batch_location = Path("data") / nfs_location.name / "semifield-developed-images" / batch_id
            else:
                batch_location = Path("data") / nfs_location.name / "semifield-upload" / batch_id
        else:
            if developed:
                batch_location = nfs_location / "semifield-developed-images" / batch_id
            else:
                batch_location = nfs_location / "semifield-upload" / batch_id
        # Check if the batch directory exists
        if batch_location.exists():
            dir_found = True
            if developed:
                files = list(Path(batch_location, "pngs").glob("*.png")) + list(
                    Path(batch_location, "pngs").glob("*.PNG"))
            else:
                files = list(batch_location.glob("*.RAW")) + list(
                    batch_location.glob("*.raw"))
            # Check if any RAW files are present
            if files:
                files_found = True
                # largest_file = max(raws_files, key=os.path.getsize)
                # Check if all RAW files have been completely uploaded
                # if all(os.path.getsize(file) == os.path.getsize(largest_file) for file in raws_files):
                # upload_complete = True
                log.info(f"Batch {batch_id} found in {batch_location} with {len(files)} {'RAW' if not developed else 'PNG'} files")
                return nfs_location
    if not dir_found:
        log.error(f"Batch {batch_id} not found in NFS locations: {nfs_locations}")
    elif not files_found:
        log.error(f"Batch {batch_id} found in {batch_location} but no RAW files found")
    elif not upload_complete:
        log.error(f"Batch {batch_id} found in {batch_location} but RAW files are not completely uploaded")
    return None

def log_image_stats(image: np.ndarray, label: str = "Image"):
    log.debug(f"{label} - dtype: {image.dtype}, range: [{np.min(image)}, {np.max(image)}], shape: {image.shape}")
