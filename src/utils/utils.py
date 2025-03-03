from pathlib import Path
import logging
import numpy as np
log = logging.getLogger(__name__)

def find_raw_dir(local_data_dir: Path, batch_id: str, lts_dir: Path) -> Path | None:
        """Find the raw directory containing .RAW files, preferring the one with more files."""
        def count_raw_files(directory: Path) -> int:
            """Return the count of .RAW files if the directory exists, otherwise 0."""
            return len(list(directory.glob("*.RAW"))) if directory.exists() else 0

        local_raw_dir = Path(local_data_dir, lts_dir.name, "semifield-upload", batch_id)
        remote_raw_dir = Path(lts_dir, "semifield-upload", batch_id)

        if not remote_raw_dir.exists():
            log.error(f"Remote RAW directory not found: {remote_raw_dir}. Exiting.")
            raise FileNotFoundError(f"Remote RAW directory not found: {remote_raw_dir}")

        local_count = count_raw_files(local_raw_dir)
        remote_count = count_raw_files(remote_raw_dir)

        if local_count > 0 or remote_count > 0:
            if local_count >= remote_count:
                log.info(f"Using local RAW directory: {local_raw_dir} ({local_count} files)")
                return local_raw_dir
            else:
                log.info(f"Using remote RAW directory: {remote_raw_dir} ({remote_count} files)")
                return remote_raw_dir

        log.warning(f"No RAW directory found for batch {batch_id}")
        return None 

# Find the batch NFS location from a list of possible parent directories
def find_lts_dir(batch_id: str, nfs_locations: list[str], local:bool=False,
                 developed:bool=False) -> Path | None:
    """
    Searches for the specified batch directory within the given NFS locations and checks for the presence and completeness of RAW files.
    Args:
        batch_id (str): The identifier of the batch to search for.
        nfs_locations (list): A list of NFS locations (directories) to search within.
        local (bool): true - searches for batch data in local directory.
        developed (bool): true - searches for pngs in semifield-developed-images, false - searches for raws in semifield-upload.
    Returns:
        Path: The NFS location where the batch was found with complete RAW
        files, or None if the batch is not found or the files are incomplete.
    Logs:
        - Info: Logs the NFS location and the number of RAW files found if the batch is found and the files are complete.
        - Error: Logs an error message if the batch is not found, if no RAW files are found, or if the RAW files are not completely uploaded.
    """
    dir_found, files_found, upload_complete = False, False, False
    batch_location = None
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
                # todo: md5 checksum for data verification?
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
