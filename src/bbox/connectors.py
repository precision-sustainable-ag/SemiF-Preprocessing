import os
import sys
from multiprocessing import Pool, cpu_count
from typing import Callable

sys.path.append("..")

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from utils.datasets import BBox, BoxCoordinates, CameraInfo, RemapImage
from tqdm import tqdm


class SfMComponents:
    def __init__(
        self,
        base_path,
        camera_reference_file="camera_reference.csv",
        fov_reference_file="fov.csv",
        gcp_reference_file="gcp_reference.csv",
    ):
        self.base_path = base_path
        self.camera_reference_file = camera_reference_file
        self.fov_reference_file = fov_reference_file
        self.gcp_reference_file = gcp_reference_file
        self._camera_reference = None
        self.dog = 34
        self._fov_reference = None
        self._gcp_reference = None

    @property
    def camera_reference(self):
        if self._camera_reference is None:
            self._camera_reference = pd.read_csv(
                os.path.join(self.base_path, self.camera_reference_file)
            )
            self._fov_reference = pd.read_csv(
                os.path.join(self.base_path, self.fov_reference_file)
            )
            self._camera_reference = self._camera_reference.merge(
                self._fov_reference, how="inner", on="label"
            )
        return self._camera_reference

    @property
    def gcp_reference(self):
        if self._gcp_reference is None:
            self._gcp_reference = pd.read_csv(
                os.path.join(self.base_path, self.gcp_reference_file)
            )
        return self._gcp_reference


class BBoxComponents:
    """Reads bounding box coordinate files and converts to BBox class"""

    def __init__(
        self,
        data_dir,
        developed_dir,
        batch_dir,
        image_dir,
        camera_reference: pd.DataFrame,
        reader: Callable,
        multiprocessing: bool,
        *args,
        **kwargs
    ):
        self.data_dir = data_dir
        self.developed_dir = Path(developed_dir)
        self.batch_dir = Path(batch_dir)
        self.image_dir = Path(image_dir)
        self.batch_id = self.batch_dir.name
        self.batch_date = self.batch_id.split("_")[1]

        self.multiprocessing = multiprocessing

        self.camera_reference = camera_reference
        self.reader = reader
        self.image_list, self.bounding_boxes = self.reader(*args, **kwargs)
        self._bboxes = dict()

        self._images = []

    def convert_bboxes(self):
        assert not self._bboxes
        for image_id, bboxes in self.bounding_boxes.items():
            boxes = []
            for bbox in bboxes:
                unique_box_id = "_".join([image_id, bbox["id"]])

                # Convert the coordinates
                top_left = np.array(bbox["top_left"])
                top_right = np.array(bbox["top_right"])
                bottom_left = np.array(bbox["bottom_left"])
                bottom_right = np.array(bbox["bottom_right"])

                box_coordinates = BoxCoordinates(
                    top_left, top_right, bottom_left, bottom_right
                )
                box = BBox(
                    bbox_id=unique_box_id,
                    image_id=image_id,
                    local_coordinates=box_coordinates,
                    cls=bbox["cls"],
                    is_normalized=bbox["is_normalized"],
                    classifier_classname=bbox["classifier_classname"],
                    classifier_confidence=bbox["classifier_confidence"],

                )
                boxes.append(box)

            self._bboxes[image_id] = boxes

    def get_fov(self, image_id):
        row = self.camera_reference[
            self.camera_reference["label"] == image_id
        ].reset_index(drop=True)

        top_left_x = float(row.loc[0, "top_left_x"])
        top_left_y = float(row.loc[0, "top_left_y"])

        bottom_left_x = float(row.loc[0, "bottom_left_x"])
        bottom_left_y = float(row.loc[0, "bottom_left_y"])

        top_right_x = float(row.loc[0, "top_right_x"])
        top_right_y = float(row.loc[0, "top_right_y"])

        bottom_right_x = float(row.loc[0, "bottom_right_x"])
        bottom_right_y = float(row.loc[0, "bottom_right_y"])

        top_left = [top_left_x, top_left_y]
        bottom_left = [bottom_left_x, bottom_left_y]
        top_right = [top_right_x, top_right_y]
        bottom_right = [bottom_right_x, bottom_right_y]

        fov = BoxCoordinates(top_left, top_right, bottom_left, bottom_right)

        return fov

    def get_camera_location(self, image_id):
        row = self.camera_reference[
            self.camera_reference["label"] == image_id
        ].reset_index(drop=True)
        camera_x = float(row.loc[0, "Estimated_X"])
        camera_y = float(row.loc[0, "Estimated_Y"])
        camera_z = float(row.loc[0, "Estimated_Z"])

        return [camera_x, camera_y, camera_z]

    def get_pixel_dims(self, image_id):
        row = self.camera_reference[
            self.camera_reference["label"] == image_id
        ].reset_index(drop=True)
        pixel_width = float(row.loc[0, "pixel_width"])
        pixel_height = float(row.loc[0, "pixel_height"])

        return pixel_width, pixel_height

    def get_orientation_angles(self, image_id):
        row = self.camera_reference[
            self.camera_reference["label"] == image_id
        ].reset_index(drop=True)
        yaw = float(row.loc[0, "Estimated_Yaw"])
        pitch = float(row.loc[0, "Estimated_Pitch"])
        roll = float(row.loc[0, "Estimated_Roll"])

        return yaw, pitch, roll

    def get_focal_length(self, image_id):
        row = self.camera_reference[
            self.camera_reference["label"] == image_id
        ].reset_index(drop=True)
        focal_length = float(row.loc[0, "f"])
        return focal_length

    @property
    def bboxes(self):
        if not self._bboxes:
            # Convert
            self.convert_bboxes()
        return self._bboxes

    def _fetch_image_metadata(self, image):
        image_id = image["id"]
        path = image["path"]
        height, width = cv2.imread(path).shape[:2]  # This is new
        fullres_path = image["fullres_path"]
        rel_path = Path(os.path.relpath(fullres_path)).parent
        fov = self.get_fov(image_id)
        camera_location = self.get_camera_location(image_id)
        pixel_width, pixel_height = self.get_pixel_dims(image_id)
        yaw, pitch, roll = self.get_orientation_angles(image_id)
        focal_length = self.get_focal_length(image_id)
        cam_info = CameraInfo(
            camera_location=camera_location,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            focal_length=focal_length,
            fov=fov,
        )

        remap_image = RemapImage(
            rel_path=rel_path,
            #  blob_home=self.data_dir.name,
            data_root=self.developed_dir.name,
            batch_id=self.batch_id,
            height=height,
            width=width,
            image_path=path,
            image_id=image_id,
            bboxes=self.bboxes[image_id],
            camera_info=cam_info,
            fullres_path=fullres_path,
        )
        # Set the full resolution height and width
        if path != fullres_path:
            _image = cv2.imread(str(fullres_path))
            fullres_height, fullres_width = _image.shape[:2]
            remap_image.set_fullres_dims(fullres_width, fullres_height)
        # Scale the boounding box coordinates to pixel space
        scale = np.array([remap_image.width, remap_image.height])

        for bbox in remap_image.bboxes:
            bbox.set_local_scale(scale)
        return remap_image

    @property
    def images(self):
        if not self._images:
            if self.multiprocessing:
                n_processes = int(cpu_count() / 2)
                with Pool(processes=n_processes) as p:
                    _images = list(
                        tqdm(
                            p.imap(self._fetch_image_metadata, self.image_list),
                            desc="Fetching Image metadata and creating RemapImage",
                            total=len(self.image_list),
                        )
                    )
                self._images = _images
            else:
                for image in tqdm(
                    self.image_list,
                    desc="Fetching Image metadata and creating RemapImage",
                ):
                    remap_image = self._fetch_image_metadata(image)
                    self._images.append(remap_image)
        return self._images
