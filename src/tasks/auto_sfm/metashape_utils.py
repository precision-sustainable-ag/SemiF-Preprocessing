import logging
from pathlib import Path
from copy import deepcopy
from typing import Callable, Tuple
from tqdm import tqdm
import yaml
import numpy as np

import Metashape as ms

from .callbacks import percentage_callback
from .dataframe import DataFrame
from .estimation import CameraStats, MarkerStats

from src.utils.utils import get_files, sanitize_time_for_path

log = logging.getLogger(__name__)


class SfM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_id = self.cfg.batch_id
        self.season = cfg.season
        self.bbot_version = str(cfg.bbot_version)
        # Directories
        self.project_path = Path(cfg.paths.proj_path)
        self.down_photos = Path(cfg.paths.down_photos)
        self.down_masks = Path(cfg.paths.down_masks)
        self.gcp_ref = Path(self.cfg.paths.gcp_ref)
        self.cam_ref = Path(self.cfg.paths.cam_ref)
        self.err_ref = Path(self.cfg.paths.err_ref)
        self.fov_ref = Path(self.cfg.paths.fov_ref)
        self.grid_dir = Path(cfg.paths.grid_dir)
        self.dem_path = Path(self.cfg.paths.dem_path)
        self.ortho_path = Path(self.cfg.paths.ortho_path)

        self.sanitized_time = sanitize_time_for_path(cfg.start_time) if cfg.start_time else ""
        self.local_inspection_dir = (
            Path(cfg.paths.batch_dir) / "inspection" / self.sanitized_time
            if self.sanitized_time else Path(cfg.paths.batch_dir) / "inspection"
        )
        self.pdf_report = self.local_inspection_dir / f"{self.batch_id}_{self.sanitized_time}_asfm_report.pdf"
        
        self.marker_file = Path(self.cfg.paths.marker_file)
        log.info(f"Using marker file: {self.marker_file.parent.name}/{self.marker_file.name} for batch {self.batch_id}")

        # Optimize camera configs
        self.opt_cam_cfg = cfg.asfm.optimize_cameras_cfg

        # align photos configs
        self.align_photos_cfg = cfg.asfm.align_photos

        # Use maskign
        self.filter_mask=self.cfg.asfm.use_masking

        # Detect markers
        self.detect_markers_cfg = self.cfg.asfm.detect_markers

        # Depth map config
        self.depth_map_cfg = self.cfg.asfm.depth_map

        # Dense cloud config
        self.dense_cloud_cfg = self.cfg.asfm.dense_cloud

        # DEM config
        self.dem_cfg = self.cfg.asfm.dem
        
        # Ortho config
        self.ortho_cfg = self.cfg.asfm.orthomosaic

        self.doc = self.load_or_create_project()
        
        self.crs, self.markerbit = self.get_crs_and_markerbit()
        
        self.num_gpus = (
            cfg.asfm.num_gpus
            if cfg.asfm.num_gpus != "all"
            else 2 ** len(ms.app.enumGPUDevices()) - 1
        )

        self.skip_first_n_images = cfg.asfm.skip_first_n_images
        self.skip_last_n_images = cfg.asfm.skip_last_n_images

        self.metashape_key = self._read_yaml(cfg.paths.pipeline_keys).get("metashape", {}).get("lic", None)
        
    def _read_yaml(self, path: Path):
        """Reads a YAML file and returns the content."""
        with open(path, 'r') as file:
            return yaml.safe_load(file)
        
    def _remove_duplicate_cameras(self, chunk: int) -> None:
        chunk_obj = self.doc.chunks[chunk]
        seen_aligned = set()
        cameras_to_remove = []

        for camera in chunk_obj.cameras:
            if camera.transform is None:
                continue
            if camera.label in seen_aligned:
                cameras_to_remove.append(camera)
            else:
                seen_aligned.add(camera.label)

        if cameras_to_remove:
            chunk_obj.remove(cameras_to_remove)

        log.info(
            f"Chunk {chunk} ({chunk_obj.label}) after duplicate cleanup: "
            f"total={len(chunk_obj.cameras)}, "
            f"unaligned={len(self.get_unaligned_cameras(chunk))}"
        )

    def remove_unaligned_cameras(self, chunk: int) -> None:
        chunk_obj = self.doc.chunks[chunk]
        unaligned = [camera for camera in chunk_obj.cameras if camera.transform is None]

        if unaligned:
            chunk_obj.remove(unaligned)

        log.info(
            f"Removed {len(unaligned)} unaligned cameras from chunk {chunk} ({chunk_obj.label}). "
            f"Remaining cameras: {len(chunk_obj.cameras)}"
        )

    def get_camera_stats(self, show=True):
        """Get the number of aligned, unaligned, and duplicate cameras for each chunk."""
        stats = []
        for chunk in self.doc.chunks:
            chunk_stats = {
                "chunk_label": chunk.label,
                "aligned_cameras": 0,
                "unaligned_cameras": 0,
                "duplicate_cameras": 0,
                "tiepoints": 0,
            }
            f = ms.TiePoints.Filter()
            f.init(chunk, criterion = ms.TiePoints.Filter.ReprojectionError)
            f.values
            chunk_stats["tiepoints"] = len(f.values) #chunk.tie_points.values

            unique_aligned_cameras = set()
            for camera in chunk.cameras:
                if camera.transform is None:
                    chunk_stats["unaligned_cameras"] += 1
                elif camera.label in unique_aligned_cameras:
                    chunk_stats["duplicate_cameras"] += 1
                else:
                    unique_aligned_cameras.add(camera.label)
                    chunk_stats["aligned_cameras"] += 1
            stats.append(chunk_stats)
            if show:
                log.info(
                f"Chunk: {chunk_stats['chunk_label']}, "
                f"Aligned Cameras: {chunk_stats['aligned_cameras']}, "
                f"Unaligned Cameras: {chunk_stats['unaligned_cameras']}, "
                f"Duplicate Cameras: {chunk_stats['duplicate_cameras']}, "
                f"Chunk tiepoints: {chunk_stats['tiepoints']}"
            )
        return stats
    
    def get_crs_and_markerbit(self) -> Tuple[str, ms.TargetType]:
        # Marker bit
        marker_bit = ms.CircularTarget14bit if "3" in self.bbot_version else ms.CircularTarget12bit
        log.info(f"Using marker bit {marker_bit} for {self.batch_id}")

        state = self.batch_id.split("_")[0]
        year = self.batch_id.split("_")[1].split("-")[0]

        # CRS logic
        crs = f"EPSG::{self.cfg.crs}"
        log.info(f"Using CRS {crs} for {self.batch_id}")

        return crs, marker_bit
    
    
    def load_or_create_project(self) -> ms.Document:
        """Opens a project if it exists or creates and saves a project

        Returns:
            ms.Document: Metashape Document containing the project
        """
        assert self.project_path.suffix == ".psx"
        if not ms.app.activated:
            ms.License().activate(self.metashape_key)
        doc = ms.Document()
        
        if self.project_path.exists():
            log.info(f"Metashape project already exisits. Opening project file.")
            # Metashape window
            doc.open(str(self.project_path), read_only=False, ignore_lock=True)
        else:
            # Create a project and save
            log.info(f"Creating new Metashape project for {self.batch_id}")
            doc.addChunk()
            doc.save(str(self.project_path))

        return doc

    def save_project(self):
        """Save the project"""
        assert self.project_path.suffix == ".psx"
        self.doc.save()

    def add_photos(self):
        """Adds a directory to the project"""
        photos = get_files(self.cfg, task="auto_sfm")
        # check of self.skip_first_n_images is an int or None
        if isinstance(self.skip_first_n_images, int):
            photos = photos[self.skip_first_n_images:]
        if isinstance(self.skip_last_n_images, int):
            photos = photos[:-self.skip_last_n_images]
        log.info(f"Adding {len(photos)} photos to the project")
        if self.doc.chunk is None:
            self.doc.addChunk()
        self.doc.chunk.crs = ms.CoordinateSystem(self.crs)
        self.doc.chunk.addPhotos(photos)

    def add_masks(self):
        """Adds masks to the cameras"""
        self.doc.chunk.generateMasks(
            path=str(self.down_masks) + "/{filename}_mask.png",
            masking_mode=ms.MaskingMode.MaskingModeFile,
            cameras=self.doc.chunk.cameras,
        )
        self.save_project()

    def remove_low_id_markers(self, chunk: int = 0, threshold: int = 300):
        """
        Remove markers with numeric labels below a threshold (e.g. < 300) for NC batches.
        """
        if not self.batch_id.startswith("NC"):
            log.info("Skipping marker filtering (not an NC batch)")
            return

        chunk_obj = self.doc.chunks[chunk]

        markers_to_remove = []

        for marker in chunk_obj.markers:
            try:
                marker_id = int(marker.label)
            except (ValueError, TypeError):
                # Skip non-numeric labels safely
                continue

            if marker_id < threshold:
                markers_to_remove.append(marker)

        log.info(f"Removing {len(markers_to_remove)} markers with ID < {threshold}")

        if markers_to_remove:
            chunk_obj.remove(markers_to_remove)

        self.save_project()
        
    def detect_markers(
        self, chunk: int = 0, progress_callback: Callable = percentage_callback
    ):
        """Detects 12 or 14 bit circular markers"""
        self.doc.chunks[chunk].detectMarkers(
            target_type=self.markerbit,
            tolerance=50,
            filter_mask=False,
            inverted=False,
            noparity=False,
            maximum_residual=5,
            minimum_size=0,
            minimum_dist=5,
            cameras=self.doc.chunks[chunk].cameras,
            progress=progress_callback,
        )
        self.save_project()

    def import_reference(self, chunk: int = 0):
        """Imports reference points"""
        self.doc.chunks[chunk].importReference(
            path=str(self.marker_file),
            format=ms.ReferenceFormatCSV,
            columns="[n|x|y|z]",
            delimiter=";",
            group_delimiters=False,
            skip_rows=1,
            ignore_labels=False,
            create_markers=True,
            threshold=0.1,
            shutter_lag=0,
            crs=ms.CoordinateSystem(self.crs)
        )
        self.save_project()

    def export_camera_reference(self):
        """Exports the reference to a CSV file."""
        # self.doc.chunk = self.doc.chunks[1]
        reference = []
        for camera in self.doc.chunk.cameras:
            stats = CameraStats(camera).to_dict()
            calibration_params = self.camera_paramters(camera)
            stats.update(calibration_params)
            # Check camera alignment
            # https://www.agisoft.com/forum/index.php?topic=6029.msg29219#msg29219
            is_aligned = camera.transform is not None
            stats.update({"Alignment": is_aligned})
            reference.append(stats)

        dataframe = DataFrame(reference, "label")
        self.camera_reference = dataframe
        dataframe.to_csv(self.cam_ref, header=True, index=False)

    def export_gcp_reference(self):
        reference = []
        for marker in self.doc.chunk.markers:
            stats = MarkerStats(marker).to_dict()
            is_detected = len(marker.projections.items()) > 0
            stats.update({"Detected": is_detected})
            reference.append(stats)

        dataframe = DataFrame(reference, "label")
        self.gcp_reference = dataframe
        dataframe.to_csv(self.gcp_ref, header=True, index=False)

    def optimize_cameras(self, progress_callback: Callable = percentage_callback):
        """Function to optimize the cameras"""
        # Disable camera locations as reference if specified in YML
        n_cameras = len(self.doc.chunk.cameras)
        for i in range(0, n_cameras):
            self.doc.chunk.cameras[i].reference.enabled = False

        self.doc.chunk.optimizeCameras(
            fit_f=self.opt_cam_cfg.fit_f,
            fit_cx=self.opt_cam_cfg.fit_cx,
            fit_cy=self.opt_cam_cfg.fit_cy,
            fit_b1=self.opt_cam_cfg.fit_b1,
            fit_b2=self.opt_cam_cfg.fit_b2,
            fit_k1=self.opt_cam_cfg.fit_k1,
            fit_k2=self.opt_cam_cfg.fit_k2,
            fit_k3=self.opt_cam_cfg.fit_k3,
            fit_k4=self.opt_cam_cfg.fit_k4,
            fit_p1=self.opt_cam_cfg.fit_p1,
            fit_p2=self.opt_cam_cfg.fit_p2,
            fit_corrections=self.opt_cam_cfg.fit_corrections,
            adaptive_fitting=self.opt_cam_cfg.adaptive_fitting,
            tiepoint_covariance=self.opt_cam_cfg.tiepoint_covariance,
            progress=progress_callback,
        )

        self.save_project()

    def get_unaligned_cameras(
        self,
        chunk: int = -1,
    ):
        unaligned_cameras = [
            camera
            for camera in self.doc.chunks[chunk].cameras
            if camera.transform is None
        ]
        return unaligned_cameras

    def reset_region(self):
        """
        Reset the region and make it much larger than the points; necessary because if points go outside the region, they get clipped when saving
        """

        self.doc.chunk.resetRegion()
        region_dims = self.doc.chunk.region.size
        region_dims[2] *= 3
        self.doc.chunk.region.size = region_dims

        return True

    def match_photos(
        self,
        progress_callback: Callable = percentage_callback,
        chunk: int = 0,
        reference_preselection=ms.ReferencePreselectionSource,
        reset_matches: bool = True,
        cameras=None,
    ):
        """Match photos in the specified chunk."""
        log.info(f"Matching photos in chunk {chunk}")

        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus

        chunk_obj = self.doc.chunks[chunk]
        if cameras is None:
            cameras = chunk_obj.cameras

        chunk_obj.matchPhotos(
            downscale=self.align_photos_cfg.downscale,
            generic_preselection=self.align_photos_cfg.generic_preselection,
            reference_preselection=self.align_photos_cfg.reference_preselection,
            reference_preselection_mode=reference_preselection,
            filter_mask=self.filter_mask,
            mask_tiepoints=True,
            filter_stationary_points=self.align_photos_cfg.filter_stationary_points,
            keypoint_limit=600000,
            keypoint_limit_per_mpx=10000,
            tiepoint_limit=200000,
            keep_keypoints=False,
            cameras=cameras,
            guided_matching=False,
            reset_matches=reset_matches,
            subdivide_task=True,
            workitem_size_cameras=20,
            workitem_size_pairs=80,
            max_workgroup_size=100,
            progress=progress_callback,
        )

        ms.app.cpu_enable = True if ms.app.gpu_mask else False
        self.save_project()

        
    def align_photos(
        self,
        progress_callback: Callable = percentage_callback,
        chunk: int = 0,
        correct: bool = False,
    ):
        """
        Align photos in the specified chunk.

        If correct=True, iteratively try to recover only the currently unaligned
        cameras in the same chunk without resetting the existing aligned solution.
        """
        log.debug(f"[{self.batch_id}] Aligning photos in chunk {chunk}")

        chunk_obj = self.doc.chunks[chunk]
        self.doc.chunk = chunk_obj

        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus

        # Initial full alignment pass
        chunk_obj.alignCameras(
            cameras=chunk_obj.cameras,
            min_image=2,
            adaptive_fitting=self.align_photos_cfg.adaptive_fitting,
            reset_alignment=True,
            subdivide_task=True,
            progress=progress_callback,
        )

        ms.app.cpu_enable = True if ms.app.gpu_mask else False
        self.save_project()

        if correct:
            prev_unaligned_count = float("inf")
            iteration = 0

            while True:
                cur_unaligned_count = len(self.get_unaligned_cameras(chunk))
                log.info(
                    f"Recovery iteration {iteration}: "
                    f"chunk={chunk}, unaligned={cur_unaligned_count}"
                )

                if cur_unaligned_count <= 2:
                    log.info("Stopping recovery because unaligned camera count is <= 2.")
                    break

                if cur_unaligned_count >= prev_unaligned_count:
                    log.warning(
                        "Stopping recovery because no further improvement was made. "
                        f"previous={prev_unaligned_count}, current={cur_unaligned_count}"
                    )
                    break

                prev_unaligned_count = cur_unaligned_count

                self._recover_unaligned_cameras_in_place(
                    chunk=chunk,
                    progress_callback=progress_callback,
                    rematch=True,
                )
                iteration += 1

        self._remove_duplicate_cameras(chunk)
        self.remove_unaligned_cameras(chunk)
        self.doc.chunk = self.doc.chunks[chunk]
        self.reset_region()
        self.save_project()

    def recover_unaligned_only(
        self,
        chunk: int = 0,
        rematch: bool = True,
        progress_callback: Callable = percentage_callback,
    ) -> None:
        """
        Recovery-only mode for an already aligned project.

        This does not perform a full realignment. It starts from the current
        chunk state and tries to recover only the currently unaligned cameras.
        """
        chunk_obj = self.doc.chunks[chunk]
        self.doc.chunk = chunk_obj

        before = len(self.get_unaligned_cameras(chunk))
        log.info(
            f"Starting recovery-only mode on chunk {chunk} ({chunk_obj.label}). "
            f"Initial unaligned cameras: {before}"
        )

        prev_unaligned_count = float("inf")
        iteration = 0

        while True:
            cur_unaligned_count = len(self.get_unaligned_cameras(chunk))
            log.info(
                f"Recovery iteration {iteration}: "
                f"chunk={chunk}, unaligned={cur_unaligned_count}"
            )

            if cur_unaligned_count == 0:
                log.info("Stopping recovery because all cameras are aligned.")
                break

            if cur_unaligned_count <= 2:
                log.info("Stopping recovery because unaligned camera count is <= 2.")
                break

            if cur_unaligned_count >= prev_unaligned_count:
                log.warning(
                    "Stopping recovery because no further improvement was made. "
                    f"previous={prev_unaligned_count}, current={cur_unaligned_count}"
                )
                break

            prev_unaligned_count = cur_unaligned_count

            self._recover_unaligned_cameras_in_place(
                chunk=chunk,
                progress_callback=progress_callback,
                rematch=rematch,
            )
            iteration += 1

        after = len(self.get_unaligned_cameras(chunk))
        log.info(
            f"Recovery-only mode finished on chunk {chunk} ({chunk_obj.label}). "
            f"Final unaligned cameras: {after}"
        )

        # self._remove_duplicate_and_unaligned_cameras(chunk)
        self.doc.chunk = self.doc.chunks[chunk]
        self.reset_region()
        self.save_project()

    def _recover_unaligned_cameras_in_place(
        self,
        chunk: int,
        progress_callback: Callable = percentage_callback,
        rematch: bool = True,
    ) -> int:
        """
        Attempt to recover unaligned cameras inside the same chunk, without creating
        rescue chunks and without resetting already aligned cameras.

        Returns:
            int: number of currently unaligned cameras after the recovery attempt.
        """
        chunk_obj = self.doc.chunks[chunk]
        self.doc.chunk = chunk_obj

        unaligned = self.get_unaligned_cameras(chunk)
        if not unaligned:
            log.info(f"No unaligned cameras found in chunk {chunk} ({chunk_obj.label}).")
            return 0

        log.info(
            f"Attempting in-place recovery for {len(unaligned)} unaligned cameras "
            f"in chunk {chunk} ({chunk_obj.label})"
        )
        for cam in unaligned:
            log.info(f"Unaligned camera: {cam.label}")

        if rematch:
            log.info("Running matchPhotos on full chunk with reset_matches=False")
            self.match_photos(
                chunk=chunk,
                progress_callback=progress_callback,
                reset_matches=False,
                cameras=chunk_obj.cameras,
            )

        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus

        log.info("Running alignCameras on unaligned cameras with reset_alignment=False")
        chunk_obj.alignCameras(
            cameras=unaligned,
            min_image=2,
            adaptive_fitting=self.align_photos_cfg.adaptive_fitting,
            reset_alignment=False,
            subdivide_task=True,
            progress=progress_callback,
        )

        ms.app.cpu_enable = True if ms.app.gpu_mask else False
        self.save_project()

        remaining = len(self.get_unaligned_cameras(chunk))
        log.info(
            f"After in-place recovery attempt, chunk {chunk} ({chunk_obj.label}) "
            f"has {remaining} unaligned cameras remaining."
        )
        return remaining



    def build_depth_map(self, progress_callback: Callable = percentage_callback):
        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus
        log.info(
            f"Number of cameras in chunk at depth map: {len(self.doc.chunk.cameras)}"
        )
        log.debug(f"Chunks names: {[chunk.label for chunk in self.doc.chunks]}")
        
        if self.depth_map_cfg.filtering_mode == "aggressive":
            filter_mode = ms.AggressiveFiltering
        elif self.depth_map_cfg.filtering_mode == "moderate":
            filter_mode = ms.ModerateFiltering
        elif self.depth_map_cfg.filtering_mode == "mild":
            filter_mode = ms.MildFiltering
        elif self.depth_map_cfg.filtering_mode.lower() == "none":
            filter_mode = ms.NoFiltering
        else:
            raise ValueError(f"Unknown filtering mode: {self.depth_map_cfg.filtering_mode}")
        
        self.doc.chunk.buildDepthMaps(
            downscale=self.depth_map_cfg.downscale,
            filter_mode=filter_mode,
            cameras=self.doc.chunk.cameras,
            reuse_depth=True,
            max_neighbors=self.depth_map_cfg.max_neighbors,
            subdivide_task=True,
            workitem_size_cameras=20,
            max_workgroup_size=100,
            progress=progress_callback,
        )
        if ms.app.gpu_mask:
            ms.app.cpu_enable = True
        
        if self.depth_map_cfg.autosave:
            self.save_project()

    def build_dense_cloud(self, progress_callback: Callable = percentage_callback):
        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus

        if self.doc.chunk.depth_maps is None:
            self.build_depth_map()

        self.doc.chunk.buildPointCloud(
            point_colors=True,
            points_spacing=self.dense_cloud_cfg.points_spacing,
            point_confidence=False,
            keep_depth=True,
            max_neighbors=100,
            uniform_sampling=True,
            subdivide_task=True,
            workitem_size_cameras=20,
            max_workgroup_size=100,
            progress=progress_callback,
        )
        if ms.app.gpu_mask:
            ms.app.cpu_enable = True

        if self.dense_cloud_cfg.autosave:
            self.save_project()

    def build_model(self, progress_callback: Callable = percentage_callback):
        self.doc.chunk.buildModel(
            surface_type=ms.HeightField,
            interpolation=ms.Extrapolated,
            face_count=ms.LowFaceCount,
            source_data=ms.PointCloudData,
            vertex_colors=True,
            vertex_confidence=False,
            volumetric_masks=False,
            keep_depth=True,
            trimming_radius=0,
            subdivide_task=True,
            workitem_size_cameras=20,
            max_workgroup_size=100,
            progress=progress_callback,
        )
        self.doc.chunk.model.closeHoles(level=100)
        self.save_project()

    def build_dem(self, progress_callback: Callable = percentage_callback):
        if self.doc.chunk.point_cloud is None:
            log.warning(f"Building dense cloud because it does not exist.")
            self.build_dense_cloud()

        self.doc.chunk.buildDem(
            source_data=ms.PointCloudData,
            interpolation=ms.EnabledInterpolation,
            flip_x=False,
            flip_y=False,
            flip_z=False,
            resolution=0,
            subdivide_task=True,
            workitem_size_tiles=10,
            max_workgroup_size=100,
            progress=progress_callback,
        )
        if self.dem_cfg.autosave:
            self.save_project()

        if self.dem_cfg.export.enabled:
            image_compression = ms.ImageCompression()
            image_compression.tiff_big = True
            kwargs = {"image_compression": image_compression}

            self.doc.chunk.exportRaster(
                path=str(self.dem_path),
                image_format=ms.ImageFormatTIFF,
                source_data=ms.ElevationData,
                progress=progress_callback,
                **kwargs,
            )

    def build_ortomosaic(self, progress_callback: Callable = percentage_callback):
        self.doc.chunk.buildOrthomosaic(
            surface_data=ms.ElevationData,
            blending_mode=ms.MosaicBlending,
            fill_holes=True,
            ghosting_filter=False,
            cull_faces=False,
            refine_seamlines=False,
            resolution=0,
            resolution_x=0,
            resolution_y=0,
            flip_x=False,
            flip_y=False,
            flip_z=False,
            subdivide_task=True,
            workitem_size_cameras=20,
            workitem_size_tiles=10,
            max_workgroup_size=100,
            progress=progress_callback,
        )

        if self.ortho_cfg.autosave:
            self.save_project()

        if self.ortho_cfg.export.enabled:
            relative_path = Path(self.ortho_path).relative_to(Path(self.cfg.paths.workdir))
            log.info(f"Exporting orthomosaic to {relative_path}")
            image_compression = ms.ImageCompression()
            image_compression.tiff_big = True

            self.doc.chunk.exportRaster(
                path=str(self.ortho_path),
                image_format=ms.ImageFormatTIFF,
                source_data=ms.OrthomosaicData,
                progress=progress_callback,
                image_compression=image_compression,
            )

    def camera_paramters(self, camera):
        row = dict()
        row["f"] = camera.calibration.f  # Focal length in pixels
        row["cx"] = camera.calibration.cx
        row["cy"] = camera.calibration.cy
        row["k1"] = camera.calibration.k1
        row["k2"] = camera.calibration.k2
        row["k3"] = camera.calibration.k3
        row["k4"] = camera.calibration.k4
        row["p1"] = camera.calibration.p1
        row["p2"] = camera.calibration.p2
        row["b1"] = camera.calibration.b1
        row["b2"] = camera.calibration.b2
        row["pixel_height"] = camera.sensor.pixel_height
        row["pixel_width"] = camera.sensor.pixel_width

        return row

    def export_stats(self):
        # Percentage of aligned images
        total_cameras = len(self.doc.chunk.cameras)
        aligned_cameras = 0
        for row in self.camera_reference.content_dict:
            aligned_cameras += row["Alignment"]
        percentage_aligned_cameras = aligned_cameras / total_cameras

        # Percentage of detected markers
        total_gcps = len(self.gcp_reference)
        detected_gcps = 0
        for row in self.gcp_reference.content_dict:
            detected_gcps += row["Detected"]
        percentage_detected_gcps = detected_gcps / max(1, total_gcps)

        dataframe = DataFrame(
            [
                {
                    "Total_Cameras": total_cameras,
                    "Aligned_Cameras": aligned_cameras,
                    "Percentage_Aligned_Cameras": percentage_aligned_cameras,
                    "Total_GCPs": total_gcps,
                    "Detected_GCPs": detected_gcps,
                    "Percentage_Detected_GCPs": percentage_detected_gcps,
                }
            ],
            "Total_Cameras",
        )
        self.error_statistics = dataframe
        dataframe.to_csv(self.err_ref, header=True, index=False)

    def camera_fov(self):
        """Calculates the field of view for each camera and saves it to a CSV file."""
        
        # Check if the chunk has a model or point cloud
        if not self.doc.chunk.shapes:
            self.doc.chunk.shapes = ms.Shapes()
            self.doc.chunk.shapes.crs = self.doc.chunk.crs
        
        # Always use the model for FOV calculation
        surface = self.doc.chunk.model
        
        # Get the transformation matrix
        transform = self.doc.chunk.transform.matrix
        
        # Get the coordinate reference system
        crs = self.doc.chunk.crs

        row_template = {
            "label": "",
            "top_left_x": "",
            "top_left_y": "",
            "bottom_left_x": "",
            "bottom_left_y": "",
            "bottom_right_x": "",
            "bottom_right_y": "",
            "top_right_x": "",
            "top_right_y": "",
            "height": "",
            "width": "",
        }
        rows = []

        for camera in tqdm(self.doc.chunk.cameras, desc="Calculating FOV", unit="camera"):
            # Skip cameras that are not regular or do not have a transform
            if camera.type != ms.Camera.Type.Regular or not camera.transform:
                continue

            # Create a row for the camera
            row = deepcopy(row_template)
            row["label"] = camera.label
            # Get the corners in pixel coordinates
            corners_px = [
                [0, 0],  # top-left
                [camera.sensor.width - 1, 0],  # top-right
                [camera.sensor.width - 1, camera.sensor.height - 1],  # bottom-right
                [0, camera.sensor.height - 1],  # bottom-left
            ]

            world_coords = []

            # Get the corners in world coordinates
            for (x, y) in corners_px:
                # Get the ray origin and target
                ray_origin = camera.unproject(ms.Vector([x, y, 0]))
                ray_target = camera.unproject(ms.Vector([x, y, 1]))

                # Pick the point on the surface
                point = surface.pickPoint(ray_origin, ray_target)
                # If no point is found, try to pick from tie points
                if point is None:
                    point = self.doc.chunk.tie_points.pickPoint(ray_origin, ray_target)
                # If still no point is found, skip this camera
                if point is None:
                    log.warning(f"Failed to get FOV corner for camera: {camera.label}")
                    break

                # Project the point to the CRS
                projected = crs.project(transform.mulp(point))  # returns Vector in CRS
                world_coords.append((projected.x, projected.y))

            if len(world_coords) != 4:
                continue  # skip this camera

            # Assign corners
            (top_left, top_right, bottom_right, bottom_left) = world_coords
            row["top_left_x"], row["top_left_y"] = top_left
            row["top_right_x"], row["top_right_y"] = top_right
            row["bottom_right_x"], row["bottom_right_y"] = bottom_right
            row["bottom_left_x"], row["bottom_left_y"] = bottom_left

            # Approximate width/height from distances
            def distance(p1, p2):
                return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
            # Calculate the average width and height in meters
            # Note: This is a rough approximation and may not be accurate for all cases
            row["width"] = (distance(top_left, top_right) + distance(bottom_left, bottom_right)) / 2.0
            row["height"] = (distance(top_left, bottom_left) + distance(top_right, bottom_right)) / 2.0

            rows.append(row)

        # Create a DataFrame and save to CSV
        df = DataFrame(rows, "label")
        df.to_csv(self.fov_ref, index=False, header=True)

    def export_pixel_world_grid(self, step: int = 100) -> None:
        """Export per-camera pixel→world coordinate grids as compressed NPZ files.

        Samples a regular pixel grid across each aligned camera and computes
        world CRS coordinates via unproject + surface pickPoint — the same
        pipeline used by camera_fov(). NPZ files are self-contained: no
        Metashape needed to consume them downstream.
        """
        surface = self.doc.chunk.model
        if surface is None:
            log.warning("No model found. Cannot export pixel-world grids.")
            return

        transform = self.doc.chunk.transform.matrix
        crs = self.doc.chunk.crs

        self.grid_dir.mkdir(parents=True, exist_ok=True)

        cameras = [
            c for c in self.doc.chunk.cameras
            if c.type == ms.Camera.Type.Regular and c.transform is not None
        ]

        log.info(f"Exporting pixel-world grids for {len(cameras)} cameras (step={step}px)")

        for camera in tqdm(cameras, desc="Exporting pixel-world grids", unit="camera"):
            w = camera.sensor.width
            h = camera.sensor.height

            # Always include sensor boundaries so interpolation covers the full frame
            u_vals = np.unique(np.concatenate([np.arange(0, w, step), [w - 1]])).astype(np.float32)
            v_vals = np.unique(np.concatenate([np.arange(0, h, step), [h - 1]])).astype(np.float32)

            world_x = np.full((len(v_vals), len(u_vals)), np.nan, dtype=np.float64)
            world_y = np.full((len(v_vals), len(u_vals)), np.nan, dtype=np.float64)
            world_z = np.full((len(v_vals), len(u_vals)), np.nan, dtype=np.float64)

            nan_count = 0
            log.info(f"Processing camera {camera.label}: sampling {len(u_vals)}x{len(v_vals)} pixels")
            for vi, v in enumerate(v_vals):
                for ui, u in enumerate(u_vals):
                    ray_origin = camera.unproject(ms.Vector([float(u), float(v), 0]))
                    ray_target = camera.unproject(ms.Vector([float(u), float(v), 1]))

                    point = surface.pickPoint(ray_origin, ray_target)
                    if point is None and self.doc.chunk.tie_points is not None:
                        point = self.doc.chunk.tie_points.pickPoint(ray_origin, ray_target)

                    if point is not None:
                        projected = crs.project(transform.mulp(point))
                        world_x[vi, ui] = projected.x
                        world_y[vi, ui] = projected.y
                        world_z[vi, ui] = projected.z
                    else:
                        nan_count += 1

            if nan_count > 0:
                log.warning(
                    f"{camera.label}: {nan_count}/{len(u_vals) * len(v_vals)} "
                    f"grid points had no surface intersection"
                )

            out_path = self.grid_dir / f"{camera.label}.npz"
            np.savez_compressed(
                str(out_path),
                u_pixels=u_vals,
                v_pixels=v_vals,
                world_x=world_x,
                world_y=world_y,
                world_z=world_z,
                sensor_width=np.array([w], dtype=np.int32),
                sensor_height=np.array([h], dtype=np.int32),
                crs=np.array([str(crs)], dtype=object),
            )
            log.debug(f"Saved grid: {out_path.name} ({len(u_vals)}x{len(v_vals)} points)")

        log.info(f"Pixel-world grid export complete: {len(cameras)} files -> {self.grid_dir}")


    def export_report(self, progress_callback: Callable = percentage_callback):
        self.doc.chunk.exportReport(
            path=str(self.pdf_report),
            title=self.batch_id,
            description="report",
            font_size=12,
            page_numbers=True,
            include_system_info=True,
            progress=progress_callback,
        )
