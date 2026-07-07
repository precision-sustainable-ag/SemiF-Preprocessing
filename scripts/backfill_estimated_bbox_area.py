import json
import logging
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.label_utils.bbox_area_estimation import (
    estimate_bbox_area_sqm,
    sensor_pixel_pitch_mm,
)
from src.utils.utils import (
    check_cam_angle,
    check_z_axis,
    extract_season_info,
    find_lts_dir,
    matches_batch_format,
)

log = logging.getLogger(__name__)

REQUIRED_CAMERA_FIELDS = ("pixel_width", "pixel_height", "focal_length", "z_axis", "cam_angle")


class MetadataAreaBackfiller:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.use_lts_metadata = bool(cfg.get("use_lts_metadata", False))
        self.scan_lts = bool(cfg.get("scan_lts", False))
        self.apply_discovered = bool(cfg.get("apply_discovered", False))
        self.target_class_id = self._optional_int(cfg.get("target_class_id"))
        self.limit_batches = self._optional_int(cfg.get("limit_batches"))
        self.scan_state = self._optional_str(cfg.get("scan_state"))
        self.scan_year = self._optional_int(cfg.get("scan_year"))

        self.sensor_width_mm = float(cfg.exif.SensorWidth)
        self.sensor_height_mm = float(cfg.exif.SensorHeight)
        self.focal_length_mm = float(cfg.exif.FocalLength)
        self.colorchecker_class_id = self._load_colorchecker_class_id()

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value in (None, "", "null"):
            return None
        return int(value)

    @staticmethod
    def _optional_str(value: Any) -> Optional[str]:
        if value in (None, "", "null"):
            return None
        return str(value).strip().upper()

    @staticmethod
    def _read_json(path: Path) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
            f.write("\n")

    def _load_colorchecker_class_id(self) -> Optional[int]:
        species_info_path = Path(self.cfg.paths.species_info)
        if not species_info_path.exists():
            return None

        data = self._read_json(species_info_path)
        for species in data.get("species", {}).values():
            if str(species.get("common_name", "")).strip().lower() == "colorchecker":
                class_id = species.get("class_id")
                return int(class_id) if class_id is not None else None
        return None

    @staticmethod
    def _is_reconstructed_batch(batch_dir: Path) -> bool:
        return (batch_dir / "reference" / "fov.csv").exists()

    @staticmethod
    def _annotation_area_missing(annotation: dict) -> bool:
        global_coordinates = annotation.get("global_coordinates") or {}
        return global_coordinates.get("area_sqm") in (None, 0, 0.0)

    @staticmethod
    def _camera_info_missing(metadata: dict) -> bool:
        camera_info = metadata.get("camera_info") or {}
        return any(camera_info.get(field) in (None, "") for field in REQUIRED_CAMERA_FIELDS)

    @staticmethod
    def _metadata_species_ids(metadata_files: list[Path]) -> set[int]:
        species_ids: set[int] = set()
        for metadata_file in metadata_files:
            data = MetadataAreaBackfiller._read_json(metadata_file)
            for ann in data.get("annotations", []):
                class_id = ann.get("category_class_id")
                if class_id is not None:
                    species_ids.add(int(class_id))
        return species_ids

    def _primary_species_id(self, species_ids: set[int]) -> Optional[int]:
        filtered = set(species_ids)
        if self.colorchecker_class_id is not None:
            filtered.discard(self.colorchecker_class_id)
        if len(filtered) != 1:
            return None
        return next(iter(filtered))

    def _batch_context(self, batch_id: str, batch_dir: Optional[Path] = None) -> dict[str, Any]:
        cfg = deepcopy(self.cfg)
        cfg.batch_id = batch_id
        cfg = extract_season_info(cfg)

        z_axis = float(check_z_axis(str(cfg.z_axis)))
        cam_angle = float(check_cam_angle(str(cfg.cam_angle)))
        bbot_version = str(cfg.bbot_version)

        if "3.1" in bbot_version:
            fullres_width = int(cfg.exif.SVCamImageWidth)
            fullres_height = int(cfg.exif.SVCamImageHeight)
        else:
            fullres_width = int(cfg.exif.SonyImageWidth)
            fullres_height = int(cfg.exif.SonyImageHeight)

        pixel_width_mm, pixel_height_mm = sensor_pixel_pitch_mm(
            self.sensor_width_mm,
            self.sensor_height_mm,
            fullres_width,
            fullres_height,
        )

        if batch_dir is None:
            if self.use_lts_metadata:
                lts_root = find_lts_dir(batch_id, cfg.paths.lts_locations, developed=True, jpgs=True)
                batch_dir = Path(lts_root) / "semifield-developed-images" / batch_id
            else:
                batch_dir = Path(cfg.paths.batch_dir)

        metadata_dir = batch_dir / "metadata"
        return {
            "cfg": cfg,
            "batch_id": batch_id,
            "batch_dir": batch_dir,
            "metadata_dir": metadata_dir,
            "bbot_version": bbot_version,
            "z_axis": z_axis,
            "cam_angle": cam_angle,
            "fullres_width": fullres_width,
            "fullres_height": fullres_height,
            "pixel_width_mm": pixel_width_mm,
            "pixel_height_mm": pixel_height_mm,
        }

    def _matches_scan_filters(self, batch_id: str) -> bool:
        state, date_str = batch_id.split("_", 1)
        year = int(date_str.split("-", 1)[0])

        if self.scan_state is not None and state != self.scan_state:
            return False
        if self.scan_year is not None and year != self.scan_year:
            return False
        return True

    def _update_camera_info(self, metadata: dict, context: dict[str, Any]) -> None:
        camera_info = metadata.setdefault("camera_info", {})
        effective_z_axis = self._coalesce_float(camera_info.get("z_axis"), context["z_axis"])
        effective_cam_angle = self._coalesce_float(camera_info.get("cam_angle"), context["cam_angle"])

        camera_info["z_axis"] = effective_z_axis
        camera_info["cam_angle"] = effective_cam_angle
        camera_info["pixel_width"] = context["pixel_width_mm"]
        camera_info["pixel_height"] = context["pixel_height_mm"]
        camera_info["focal_length"] = self.focal_length_mm
        camera_coefficients = camera_info.setdefault("camera_coefficients", {})
        camera_coefficients["f"] = self.focal_length_mm

    @staticmethod
    def _coalesce_float(existing_value: Any, fallback_value: float) -> float:
        if existing_value in (None, "", "null"):
            return float(fallback_value)
        return float(existing_value)

    def _update_annotation(self, annotation: dict, context: dict[str, Any]) -> bool:
        bbox_xywh = annotation.get("bbox_xywh")
        if not bbox_xywh:
            return False

        annotation.pop("local_coordinates", None)

        global_coordinates = annotation.get("global_coordinates") or {}
        camera_info = context["metadata"].get("camera_info") or {}
        effective_z_axis = self._coalesce_float(camera_info.get("z_axis"), context["z_axis"])
        effective_cam_angle = self._coalesce_float(camera_info.get("cam_angle"), context["cam_angle"])
        area_sqm = estimate_bbox_area_sqm(
            bbox_xywh=bbox_xywh,
            pixel_width_mm=context["pixel_width_mm"],
            pixel_height_mm=context["pixel_height_mm"],
            focal_length_mm=self.focal_length_mm,
            z_axis_cm=effective_z_axis,
            cam_angle_deg=effective_cam_angle,
        )
        if area_sqm is not None:
            global_coordinates["area_sqm"] = area_sqm
            annotation["global_coordinates"] = global_coordinates
            return True

        annotation["global_coordinates"] = global_coordinates
        return False

    def _apply_backfill(self, context: dict[str, Any]) -> tuple[int, int]:
        metadata_files = sorted(context["metadata_dir"].glob("*.json"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata JSON files found for batch {context['batch_id']}")

        updated_files = 0
        updated_annotations = 0

        log.info(
            "Applying backfill for %s using metadata at %s (%s files)",
            context["batch_id"],
            context["metadata_dir"],
            len(metadata_files),
        )

        for metadata_file in tqdm(
            metadata_files,
            desc=f"Backfill {context['batch_id']}",
            unit="file",
            leave=False,
        ):
            metadata = self._read_json(metadata_file)
            context["metadata"] = metadata
            self._update_camera_info(metadata, context)

            file_updates = 0
            for annotation in metadata.get("annotations", []):
                if self._update_annotation(annotation, context):
                    file_updates += 1

            self._write_json(metadata_file, metadata)
            updated_files += 1
            updated_annotations += file_updates
            log.info("Updated %s annotations in %s", file_updates, metadata_file)

        return updated_files, updated_annotations

    def _scan_lts_batches(self) -> list[dict[str, Any]]:
        discovered: dict[str, Path] = {}
        for lts_root in self.cfg.paths.lts_locations:
            developed_root = Path(lts_root) / "semifield-developed-images"
            if not developed_root.exists():
                continue
            log.info("Scanning LTS root: %s", developed_root)
            for batch_dir in sorted(developed_root.iterdir()):
                if not batch_dir.is_dir() or not matches_batch_format(batch_dir.name):
                    continue
                if not self._matches_scan_filters(batch_dir.name):
                    continue
                discovered.setdefault(batch_dir.name, batch_dir)

        eligible = []
        log.info("Discovered %s candidate batches before eligibility checks.", len(discovered))
        for batch_id, batch_dir in tqdm(
            sorted(discovered.items()),
            desc="Evaluate LTS batches",
            unit="batch",
            leave=False,
        ):
            metadata_dir = batch_dir / "metadata"
            if not metadata_dir.exists():
                continue
            if self._is_reconstructed_batch(batch_dir):
                continue

            metadata_files = sorted(metadata_dir.glob("*.json"))
            if not metadata_files:
                continue

            species_ids = self._metadata_species_ids(metadata_files)
            primary_species_id = self._primary_species_id(species_ids)
            if primary_species_id is None:
                continue

            if self.target_class_id is not None and primary_species_id != self.target_class_id:
                continue

            missing_area = False
            missing_camera = False
            for metadata_file in metadata_files:
                data = self._read_json(metadata_file)
                if self._camera_info_missing(data):
                    missing_camera = True
                if any(self._annotation_area_missing(ann) for ann in data.get("annotations", [])):
                    missing_area = True
                if missing_area or missing_camera:
                    break

            if not (missing_area or missing_camera):
                continue

            context = self._batch_context(batch_id=batch_id, batch_dir=batch_dir)
            eligible.append({
                "batch_id": batch_id,
                "batch_dir": batch_dir,
                "metadata_dir": metadata_dir,
                "species_id": primary_species_id,
                "all_class_ids": sorted(species_ids),
                "missing_area": missing_area,
                "missing_camera": missing_camera,
                "context": context,
            })

        if self.limit_batches is not None:
            eligible = eligible[:self.limit_batches]
        log.info("Eligible batches after filtering: %s", len(eligible))
        return eligible

    def _print_scan_summary(self, eligible: list[dict[str, Any]]) -> None:
        if not eligible:
            log.info("No eligible LTS batches found.")
            return

        grouped: dict[int, list[str]] = defaultdict(list)
        for row in eligible:
            grouped[row["species_id"]].append(row["batch_id"])

        log.info("Eligible non-reconstructed LTS batches requiring backfill:")
        for species_id, batch_ids in sorted(grouped.items()):
            log.info("  class_id=%s -> %s batches", species_id, len(batch_ids))
            for batch_id in batch_ids:
                log.info("    %s", batch_id)

    def run_single_batch(self) -> None:
        context = self._batch_context(batch_id=self.cfg.batch_id)
        updated_files, updated_annotations = self._apply_backfill(context)
        log.info(
            "Backfill complete for %s: %s files updated, %s annotations received estimated area.",
            self.cfg.batch_id,
            updated_files,
            updated_annotations,
        )

    def run_scan(self) -> None:
        eligible = self._scan_lts_batches()
        self._print_scan_summary(eligible)

        if not self.apply_discovered:
            return

        if not eligible:
            log.info("No eligible batches to apply.")
            return

        for row in eligible:
            log.info(
                "Applying discovered batch %s (class_id=%s, all_class_ids=%s)",
                row["batch_id"],
                row["species_id"],
                row["all_class_ids"],
            )
            updated_files, updated_annotations = self._apply_backfill(row["context"])
            log.info(
                "Applied backfill to %s: %s files updated, %s annotations updated.",
                row["batch_id"],
                updated_files,
                updated_annotations,
            )

    def run(self) -> None:
        if self.scan_lts:
            self.run_scan()
        else:
            self.run_single_batch()


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    backfiller = MetadataAreaBackfiller(cfg)
    backfiller.run()


if __name__ == "__main__":
    main()
