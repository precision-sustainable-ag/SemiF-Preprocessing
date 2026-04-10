import logging
import shutil
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.utils.utils import find_lts_dir, is_reconstructed, get_files

log = logging.getLogger(__name__)

class CleanUpLocalTemp:
    """Move final batch outputs to LTS and clean local temporary data."""
    def __init__(self, cfg, batch_id: str):
        self.cfg = cfg
        self.batch_id = batch_id

        self.local_batch_dir = Path(cfg.paths.batch_dir)
        self.lts_batch_dir = (
            find_lts_dir(
                batch_id,
                cfg.paths.lts_locations,
                local=False,
                developed=True,
                dngs=False,
                jpgs=True,
            )
            / "semifield-developed-images"
            / batch_id
        )
        self.setup_src_dst_paths()
        self.is_reconstructed = is_reconstructed(cfg)

    def setup_src_dst_paths(self):
        # Source paths
        self.src_metadata = self.local_batch_dir / "metadata"
        self.src_cam_references = Path(self.cfg.paths.refs)
        self.src_log_path = Path(HydraConfig.get().runtime.output_dir) / f"{self.cfg.batch_id}.log"

        configured_inspection_dir = Path(self.cfg.paths.inspection_dir)

        if configured_inspection_dir.exists() and configured_inspection_dir.is_dir():
            self.src_inspection_dir = configured_inspection_dir
        else:
            self.src_inspection_dir = self.local_batch_dir / "inspection"

        # Destination paths
        self.dst_metadata = self.lts_batch_dir / "metadata"
        self.dst_metadata.mkdir(parents=True, exist_ok=True)

        self.dst_cam_references = self.lts_batch_dir / "reference"
        self.dst_cam_references.mkdir(parents=True, exist_ok=True)

        self.dst_inspection_dir = self.lts_batch_dir / "inspection"
        self.dst_inspection_dir.mkdir(parents=True, exist_ok=True)

    def _relative_files(self, root: Path) -> set[str]:
        if not root.exists():
            return set()
        return {
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file()
        }

    def _copy_all_files_overwrite(self, src_root: Path, dst_root: Path):
        """Copy all files from src_root to dst_root, preserving structure and overwriting existing files."""
        if not src_root.exists():
            log.warning(f"Source directory does not exist: {src_root}")
            return

        file_count = 0
        for src_path in src_root.rglob("*"):
            if not src_path.is_file():
                continue

            rel_path = src_path.relative_to(src_root)
            dst_path = dst_root / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src_path, dst_path)
            file_count += 1

        log.info(
            f"Copied {file_count} files from {src_root} to {dst_root} "
            f"(existing files overwritten)."
        )

    def _inspection_copy_verified(self) -> bool:
        src_files = self._relative_files(self.src_inspection_dir)
        dst_files = self._relative_files(self.dst_inspection_dir)

        missing = src_files - dst_files
        if missing:
            log.error(
                f"Inspection files missing in LTS directory {self.dst_inspection_dir}: "
                f"{len(missing)}"
            )
            for rel_path in sorted(list(missing))[:20]:
                log.error(f"Missing inspection file: {rel_path}")
            return False
        return True

    def move_data(self):
        """Move the metadata and inspection results to the LTS directory."""
        if self.src_metadata.exists():
            shutil.copytree(str(self.src_metadata), str(self.dst_metadata), dirs_exist_ok=True)
            log.info(f"Copied {self.src_metadata} to {self.dst_metadata}")
        else:
            log.warning(f"Metadata directory {self.src_metadata} does not exist. Skipping move.")

        if self.src_cam_references.exists():
            shutil.copytree(str(self.src_cam_references), str(self.dst_cam_references), dirs_exist_ok=True)
            log.info(f"Copied {self.src_cam_references} to {self.dst_cam_references}")
        else:
            log.warning(f"Camera references directory {self.src_cam_references} does not exist. Skipping move.")

        if self.src_inspection_dir.exists():
            log.info(
                f"Resolved inspection source exists={self.src_inspection_dir.exists()} "
                f"is_dir={self.src_inspection_dir.is_dir()}"
            )
            log.info(f"Inspection source: {self.src_inspection_dir}")
            log.info(f"Inspection destination: {self.dst_inspection_dir}")

            for p in sorted(self.src_inspection_dir.iterdir()):
                log.info(f"Inspection child: {p} | is_dir={p.is_dir()} | is_file={p.is_file()}")

            if self.src_log_path.exists():
                dst_log_in_inspection = self.src_inspection_dir / self.src_log_path.name
                shutil.copy2(str(self.src_log_path), str(dst_log_in_inspection))
                log.info(f"Copied log file {self.src_log_path} to {dst_log_in_inspection}")

            shutil.copytree(
                str(self.src_inspection_dir),
                str(self.dst_inspection_dir),
                dirs_exist_ok=True
            )
            log.info(f"Copied {self.src_inspection_dir} to {self.dst_inspection_dir}")
        else:
            log.warning(f"Inspection directory {self.src_inspection_dir} does not exist. Skipping move.")

    def can_remove_local_dir(self):
        """Check that all the files in the temp directories are in the LTS directories."""
        can_remove_local_dir = False

        temp_metadata = get_files(self.cfg, task="move_data_temp_data")
        lts_metadata = get_files(self.cfg, task="move_data_lts_data")

        lts_cam_references_exists = len(list(self.dst_cam_references.glob("*.csv"))) == 5
        lts_inspection_results_exists = self._inspection_copy_verified()

        metadata_diff = len(temp_metadata) - len(lts_metadata)

        if self.is_reconstructed:
            if metadata_diff > 0 or not lts_cam_references_exists or not lts_inspection_results_exists:
                if metadata_diff:
                    log.error(f"Metadata files in local batch directory not found in LTS directory: {metadata_diff}")
                if not lts_cam_references_exists:
                    log.error(f"Camera references not found in LTS directory: {self.dst_cam_references}")
                if not lts_inspection_results_exists:
                    log.error(f"Inspection results not fully found in LTS directory: {self.dst_inspection_dir}")
                log.error(f"Local batch directory {self.local_batch_dir} cannot be removed.")
                can_remove_local_dir = False
            else:
                log.info(f"All files in temp directories are in the LTS directories for batch {self.batch_id}.")
                can_remove_local_dir = True
        else:
            if metadata_diff > 0 or not lts_inspection_results_exists:
                if metadata_diff:
                    log.error(f"Metadata files in local batch directory not found in LTS directory: {metadata_diff}")
                if not lts_inspection_results_exists:
                    log.error(f"Inspection results not fully found in LTS directory: {self.dst_inspection_dir}")
                log.error(f"Local batch directory {self.local_batch_dir} cannot be removed.")
                can_remove_local_dir = False
            else:
                log.info(f"All files in temp directories are in the LTS directories for batch {self.batch_id}.")
                can_remove_local_dir = True

        return can_remove_local_dir

    def cleanup_temp(self):
        """Remove the batch_id folder from the temp directory."""
        self.move_data()
        log.info(f"Moved data for batch {self.batch_id} to LTS directory.")

        can_remove = self.can_remove_local_dir()

        if not can_remove:
            log.error(f"Cannot remove temp directories for batch {self.batch_id}. Exiting.")
            return

        try:
            local_batch_dir_contents = self.local_batch_dir.glob("*")
            for item in local_batch_dir_contents:
                if item.is_dir() and item.name != "inspection":
                    shutil.rmtree(item)
                    log.info(f"Removed {item}.")
                elif item.is_file():
                    item.unlink()
                    log.info(f"Removed {item}.")

            prediction_images = self.local_batch_dir / "inspection" / "prediction_images"
            if prediction_images.exists():
                shutil.rmtree(prediction_images)
                log.info(f"Removed {prediction_images}.")

        except Exception as e:
            log.error(f"Failed to remove temp directories for batch {self.batch_id}: {e}")
            return

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log.info("Starting cleanup of local temp directories.")
    batch_id = cfg.batch_id
    try:
        cleaner = CleanUpLocalTemp(cfg, batch_id)
        cleaner.cleanup_temp()
        log.info("Finished cleanup of local temp directories.")
    except Exception as e:
        log.error(f"Error during cleanup: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()