import logging
from typing import Dict, List, Type

import Metashape
import numpy as np
from utils.datasets import BBox, BoxCoordinates, ImageData
from tqdm import tqdm

from .bbox_utils import bb_iou, generate_hash

log = logging.getLogger(__name__)

FOV_IOU_THRESH = 0.1
BBOX_OVERLAP_THRESH = 0.3


class BBoxFilter:
    def __init__(self, images: List[ImageData]):
        self.images = images
        self.image_map = {image.image_id: image for image in images}
        self.total_bboxes = sum([len(image.bboxes) for image in self.images])
        self.primary_boxes = []
        self.primary_box_ids = set()

    def deduplicate_bboxes(self):
        """Calculates the ideal bounding box and the associated image from all the
        bounding boxes
        """
        comparisons = self.filter_images()
        self.filter_bounding_boxes(comparisons)

    def filter_images(self) -> Dict[str, List[str]]:
        """Filter the images to compare based on the overlap between their fields of
           view

        Returns:
            Dict[str, List[str]]: A dictionary containing the image IDs as keys, and
                                  a list of image IDs each key overlaps with
        """
        image_ids = list(self.image_map.keys())
        comparisons = dict()
        # Find the overlap between FOVs of the images
        for i, image_id in enumerate(image_ids):
            image = self.image_map[image_id]
            comparisons[image_id] = []
            for j in range(i + 1, len(image_ids)):
                compare_image_id = image_ids[j]
                compare_image = self.image_map[compare_image_id]
                fov_iou = bb_iou(image.camera_info.fov, compare_image.camera_info.fov)
                if fov_iou > FOV_IOU_THRESH:
                    comparisons[image_id].append(compare_image_id)

        return comparisons

    def filter_bounding_boxes(self, comparisons: Dict[str, List[str]]):
        """Find overlapping bounding boxes from the images to compare

        Args:
            comparisons (Dict[str, List[str]]): Images to compare, found via
                                                filter_images
        """
        # For all the overlapping images
        visited_bboxes = set()
        areas = []

        for image_id, image_ids_for_comparison in comparisons.items():
            # For each bounding box in the key image
            for box in self.image_map[image_id].bboxes:
                if box.bbox_id not in visited_bboxes:
                    visited_bboxes.add(box.bbox_id)
                    areas.append(box.local_area)
                compared = set()
                # A unique ID for the bounding box in question
                box_hash = generate_hash(box)
                # For each overlapping image
                for compare_image_id in image_ids_for_comparison:
                    boxes = self.image_map[compare_image_id].bboxes
                    # And each of its bounding box
                    for _box in boxes:
                        if _box.bbox_id not in visited_bboxes:
                            visited_bboxes.add(_box.bbox_id)
                            areas.append(_box.local_area)

                        # A unique ID for a pair of bounding boxes
                        # Note that the order of the boxes does not matter
                        # i.e. Box_A,Box_B is the same as Box_B,Box_A
                        _box_hash = generate_hash(_box, box_hash)
                        if _box_hash in compared:
                            continue
                        compared.add(_box_hash)
                        iou = box.bb_iou(_box)
                        if iou > BBOX_OVERLAP_THRESH:
                            # Set the two boxes as overlapping
                            box.add_box(_box)
                            _box.add_box(box)

        self.select_best_bbox()
        self.cleanup_primary_boxes()

    def select_best_bbox(self):
        # visited will be a set of boxes that have been compared
        visited = set()
        for image in self.images:
            # If all the boxes have been checked, no need to
            # check the other images
            if len(visited) == self.total_bboxes:
                break
            bboxes = image.bboxes
            for box in bboxes:
                box_hash = generate_hash(box)
                if box_hash in visited:
                    continue
                all_boxes = [box] + box._overlapping_bboxes
                box_hashes = [generate_hash(_box) for _box in all_boxes]
                visited = visited.union(set(box_hashes))
                # Find the best bounding box
                centers = np.array(
                    [
                        self.image_map[_box.image_id].camera_info.camera_location
                        for _box in all_boxes
                    ]
                )
                centroids = np.array([_box.global_centroid for _box in all_boxes])
                distances = 0
                try:
                    distances = ((centroids - centers[:, :2]) ** 2).sum(axis=-1)
                except ValueError as e:
                    log.exception(f"Error calculating distances: {str(e)}")
                    log.error(f"Centroids: {centroids}")
                    log.error(f"Centers: {centers}")
                    log.error(f"Centers [:, :2]: {centers[:, :2]}")
                    continue
                min_idx = np.argmin(distances)
                all_boxes[min_idx].is_primary = True
                if all_boxes[min_idx].bbox_id not in self.primary_box_ids:
                    self.primary_boxes.append(all_boxes[min_idx])
                    self.primary_box_ids.add(all_boxes[min_idx].bbox_id)

    def cleanup_primary_boxes(self):
        _primary_boxes = []
        for i, box in enumerate(self.primary_boxes):
            image_width = self.image_map[box.image_id].width
            image_height = self.image_map[box.image_id].height

            if (
                box.local_centroid[0] < image_width // 4
                or box.local_centroid[0] > 3 * image_width // 4
                or box.local_centroid[1] < image_height // 4
                or box.local_centroid[1] > 3 * image_height // 4
            ):
                box.is_primary = False
                # del self.primary_boxes[i]
            else:
                _primary_boxes.append(box)

        # Revisit all bounding boxes identified as primary and
        # remove the overlapping ones
        for i in range(len(_primary_boxes)):
            box1 = _primary_boxes[i]
            camera_location1 = self.image_map[
                box1.image_id
            ].camera_info.camera_location[
                :2
            ]  # get just x and y
            for j in range(i + 1, len(_primary_boxes)):
                box2 = _primary_boxes[j]
                camera_location2 = self.image_map[
                    box2.image_id
                ].camera_info.camera_location[
                    :2
                ]  # get just x and y
                iou = box1.bb_iou(box2)
                if iou > BBOX_OVERLAP_THRESH:
                    # De-duplicate
                    distance1 = ((box1.global_centroid - camera_location1) ** 2).sum(
                        axis=-1
                    )
                    distance2 = ((box2.global_centroid - camera_location2) ** 2).sum(
                        axis=-1
                    )

                    if distance1 < distance2:
                        box2.is_primary = False
                    else:
                        box1.is_primary = False


class BBoxMapper:
    def __init__(self, project_path: str, images: List[ImageData]):
        """Class to map bounding box coordinates from image cordinates
        to global coordinates
        """
        self.images = images
        self.doc = Metashape.Document()
        self.doc.open(str(project_path), ignore_lock=True)

    def map(self):
        """
        Maps all the bounding boxes to a global coordinate space
        """

        # Create a list of chunks for each image_id
        image_id_to_chunks = {image.image_id: [] for image in self.images}
        chunck_labels = [chunk.label for chunk in self.doc.chunks]
        if "Merged Chunk" in chunck_labels:
            chunk = self.doc.chunks[-1]
            log.info(f"Using 'Merged Chunk' for processing. {chunk}")
            # for chunk in self.doc.chunks:
            for camera in chunk.cameras:
                # print(camera.label)
                if camera.label in image_id_to_chunks:
                    image_id_to_chunks[camera.label].append(chunk)
        elif (len(chunck_labels) == 1) and ("Chunk 1" == chunck_labels[0]):
            chunk = self.doc.chunks[0]
            log.info(f"Using 'Chunk 1' for processing. {chunk}")
            # for chunk in self.doc.chunks:
            for camera in chunk.cameras:
                # print(camera.label)
                if camera.label in image_id_to_chunks:
                    image_id_to_chunks[camera.label].append(chunk)

        else:
            for chunk in self.doc.chunks:
                for camera in chunk.cameras:
                    if camera.label in image_id_to_chunks:
                        image_id_to_chunks[camera.label].append(chunk)

        surface = chunk.model
        for image in tqdm(self.images, desc="Mapping: Projecting 2D to 3D"):
            image_id = image.image_id

            chunks = image_id_to_chunks.get(image_id, [])
            if not chunks:
                log.critical(f"No chunks found for image_id {image_id}. Skipping...")
                continue

            for bbox in image.bboxes:
                global_coordinates = None
                # for chunk in chunks:
                try:
                    global_coordinates = self.map_local_to_global(
                        bbox, image_id, chunk, surface
                    )
                    if global_coordinates:
                        bbox.update_global_coordinates(global_coordinates)
                        # break
                except Exception as e:
                    global_coordinates = BoxCoordinates(
                        np.array((0,0)),
                        np.array((0,0)),
                        np.array((0,0)),
                        np.array((0,0))
                        )
                    
                    bbox.update_global_coordinates(global_coordinates)
                    log.warning(
                        f"Error mapping bbox for image_id {image_id} using chunk {chunk}. Reason: {str(e)}\nGlobal coordinates: {global_coordinates}."
                    )
                if not global_coordinates:
                    log.critical(
                        f"Failed to map bbox for image_id {image_id} using any chunk."
                    )

    def map_local_to_global(self, bbox, image_id, chunk, surface):
        camera = next((cam for cam in chunk.cameras if cam.label == image_id), None)
        if not camera:
            raise ValueError(
                f"No camera found for image_id {image_id} in chunk {chunk}."
            )

        # surface = chunk.model
        mapped_coordinates = []

        for co, coords in zip(
            ["top_left", "bottom_left", "top_right", "bottom_right"],
            [
                bbox.local_coordinates.top_left,
                bbox.local_coordinates.bottom_left,
                bbox.local_coordinates.top_right,
                bbox.local_coordinates.bottom_right,
            ],
        ):
            x_coord, y_coord = max(coords[0], 0), max(coords[1], 0)

            ray_origin = camera.center
            if ray_origin is None:
                log.critical(f"Camera center is None. Image ID is {image_id}.")

            coords_2D = Metashape.Vector([x_coord, y_coord])
            ray_target = camera.unproject(coords_2D)
            point_internal = surface.pickPoint(ray_origin, ray_target)

            if point_internal is None:
                log.warning(
                    f"Point internal is None. Image ID is {image_id}. Ray origin is {ray_origin}. Ray target is {ray_target}. 2D coordinates {co}: {coords_2D}, the missing coordinate will used based on the other present coordinates."
                )

                # global_coord = np.array([0, 0])
                raise ValueError(
                    f"Point internal is None. Image ID is {image_id}. Ray target is {ray_target}"
                )

            else:
                global_coord = chunk.crs.project(
                    chunk.transform.matrix.mulp(point_internal)
                )[:2]
            mapped_coordinates.append(global_coord)

        bbox_coordinates = BoxCoordinates(
            np.array(mapped_coordinates[0]),
            np.array(mapped_coordinates[2]),
            np.array(mapped_coordinates[1]),
            np.array(mapped_coordinates[3]),
        )

        return bbox_coordinates

    def map_og(self):
        ### Deprecated
        """
        Maps all the bounding boxes to a global coordinate space
        """

        for image in tqdm(self.images, desc="Mapping: Projecting 2D to 3D"):
            image_id = image.image_id

            # Isolate the chunk
            camera_chunk = None
            if len(self.doc.chunks) == 3:  # using the "Merged" chunk
                log.info(
                    f"Using the 'Merged' chunk to transform pixel bbox coordinates ({self.doc.chunks[-1]})."
                )
                camera_chunk = self.doc.chunks[-1]
                cameras = [camera.label for camera in camera_chunk.cameras]
                assert image_id in cameras, f"image_id not in cameras."
                camera = [cam for cam in camera_chunk.cameras if cam.label == image_id][
                    0
                ]

            else:  # If no "Merged" chunk, finding the chunk with the image_id
                log.info(f"No 'Merged' chunk.")
                for chunk in self.doc.chunks:
                    cameras = [camera.label for camera in chunk.cameras]
                    if image_id in cameras:
                        camera_chunk = chunk
                        log.info(f"Using chunk: {camera_chunk}.")
                        camera = [
                            cam for cam in chunk.cameras if cam.label == image_id
                        ][0]
                        break

            assert camera_chunk is not None

            # From: https://www.agisoft.com/forum/index.php?topic=13875.0
            # surface = camera_chunk.point_cloud
            # surface = self.doc.chunks[-1].model
            surface = camera_chunk.model

            global_coordinates = dict()
            for bbox in image.bboxes:
                top_left = bbox.local_coordinates.top_left
                bottom_left = bbox.local_coordinates.bottom_left
                top_right = bbox.local_coordinates.top_right
                bottom_right = bbox.local_coordinates.bottom_right

                mapped_coordinates = []

                co_type = ["top_left", "bottom_left", "top_right", "bottom_right"]

                for co, coords in zip(
                    co_type, [top_left, bottom_left, top_right, bottom_right]
                ):
                    x_coord = coords[0]
                    y_coord = coords[1]

                    if x_coord < 0:
                        x_coord == 0
                    if y_coord < 0:
                        y_coord == 0

                    ray_origin = camera.center
                    if ray_origin is None:
                        log.critical(
                            f"Camera center is {ray_origin}. Image ID is {image_id}."
                        )
                    coords_2D = Metashape.Vector([x_coord, y_coord])

                    ray_target = camera.unproject(coords_2D)
                    point_internal = surface.pickPoint(ray_origin, ray_target)

                    if point_internal is None:
                        log.critical(
                            f"Point internal is {point_internal}. Image ID is {image_id}. Ray target is {ray_target}"
                        )
                        raise TypeError()
                    else:
                        # From https://www.agisoft.com/forum/index.php?topic=12781.0
                        global_coord = camera_chunk.crs.project(
                            camera_chunk.transform.matrix.mulp(point_internal)
                        )[:2]
                    mapped_coordinates.append(global_coord)

                top_left = np.array(mapped_coordinates[0])
                top_right = np.array(mapped_coordinates[2])
                bottom_left = np.array(mapped_coordinates[1])
                bottom_right = np.array(mapped_coordinates[3])

                global_coordinates = BoxCoordinates(
                    top_left, top_right, bottom_left, bottom_right
                )
                bbox.update_global_coordinates(global_coordinates)


class GlobalToLocalMaper:
    def __init__(self, project_path: str, images: List[ImageData]):
        """Class to map bounding box coordinates from global cordinates
        to image coordinates
        """
        self.images = images
        self.doc = Metashape.Document()
        self.doc.open(str(project_path))

    def map(self, global_box_coordinates: BBox, map_to: str):
        local_coordinates = self.bbox_to_local(global_box_coordinates, map_to)
        return local_coordinates

    def bbox_to_local(self, global_bbox: BBox, map_to_image: ImageData):
        assert not global_bbox.local_coordinates, (
            "The global box contains existing local coordinates"
            "Mapping operation will overwrite them."
        )

        image_id = map_to_image.image_id

        # Isolate the chunk
        camera_chunk = None
        for chunk in self.doc.chunks:
            cameras = [camera.label for camera in chunk.cameras]
            if image_id in cameras:
                camera_chunk = chunk
                camera = [cam for cam in chunk.cameras if cam.label == image_id][0]
                break

        assert camera_chunk is not None

        # From: https://www.agisoft.com/forum/index.php?topic=13875.0
        crs = camera_chunk.crs
        T = camera_chunk.transform.matrix

        top_left = global_bbox.global_coordinates.top_left
        bottom_left = global_bbox.global_coordinates.bottom_left
        top_right = global_bbox.global_coordinates.top_right
        bottom_right = global_bbox.global_coordinates.bottom_right

        mapped_coordinates = dict()

        co_type = ["top_left", "bottom_left", "top_right", "bottom_right"]

        scale = np.array([map_to_image.width, map_to_image.height])

        for co, coords in zip(
            co_type, [top_left, bottom_left, top_right, bottom_right]
        ):
            x_coord = coords[0]
            y_coord = coords[1]
            z_coord = 0.0  # Assuming 0 height. This could be tuned further

            point = Metashape.Vector(
                [x_coord, y_coord, z_coord]
            )  # point of interest in geographic coord
            point_internal = T.inv().mulp(crs.unproject(point))
            coords_2D = camera.project(point_internal)
            # Normalize
            mapped_coordinates[co] = np.array(coords_2D) / scale

        box_coordinates = BoxCoordinates(
            top_left=mapped_coordinates["top_left"],
            top_right=mapped_coordinates["top_right"],
            bottom_left=mapped_coordinates["bottom_left"],
            bottom_right=mapped_coordinates["bottom_right"],
        )

        return box_coordinates
