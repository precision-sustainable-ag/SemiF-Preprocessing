import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union
from shapely.geometry import Polygon
import cv2
import exifread
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


SCHEMA_VERSION = "2.0"


@dataclass
class BoxCoordinates:
    top_left: np.ndarray
    top_right: np.ndarray
    bottom_left: np.ndarray
    bottom_right: np.ndarray
    is_scaleable: bool = field(init=True, default=True)

    def __bool__(self):
        # The bool function is to check if the coordinates are populated or not
        return all(
            [
                len(coord) == 2
                for coord in [
                    self.top_left,
                    self.top_right,
                    self.bottom_left,
                    self.bottom_right,
                ]
            ]
        )

    @property
    def config(self):
        if isinstance(self.top_left, np.ndarray):
            _top_left = self.top_left.tolist()
            _top_right = self.top_right.tolist()
            _bottom_left = self.bottom_left.tolist()
            _bottom_right = self.bottom_right.tolist()
        else:
            _top_left = self.top_left
            _top_right = self.top_right
            _bottom_left = self.bottom_left
            _bottom_right = self.bottom_right

        _config = {
            "top_left": _top_left,
            "top_right": _top_right,
            "bottom_left": _bottom_left,
            "bottom_right": _bottom_right,
        }

        return _config

    def set_scale(self, new_scale: np.ndarray):
        if not self.is_scaleable:
            raise ValueError("is_scalable set to False, coordinates cannot be scaled.")
        self.scale = new_scale

        self.top_left = self.top_left * self.scale
        self.top_right = self.top_right * self.scale
        self.bottom_left = self.bottom_left * self.scale
        self.bottom_right = self.bottom_right * self.scale

    def copy(self):
        return self.__class__(
            top_left=self.top_left.copy(),
            top_right=self.top_right.copy(),
            bottom_left=self.bottom_left.copy(),
            bottom_right=self.bottom_right.copy(),
            is_scaleable=self.is_scaleable,
        )

    def __getitem__(self, key):
        if not hasattr(self, key):
            raise AttributeError(f"{self.__class__.__name__} has not attribute {key}")

        return getattr(self, key)


def init_empty():
    empty_array = np.array([])
    # Initialize with an empty array
    return BoxCoordinates(
        empty_array, empty_array, empty_array, empty_array, empty_array
    )


@dataclass
class BBox:
    bbox_id: str
    image_id: str
    cls: str
    classifier_classname: str
    classifier_confidence: float
    cutout_exists: Optional[bool] = field(default=False)
    instance_rgb_id: List[int] = field(default=None)
    local_coordinates: BoxCoordinates = field(init=True, default_factory=init_empty)
    global_coordinates: BoxCoordinates = field(init=True, default_factory=init_empty)
    is_normalized: bool = field(init=True, default=False)
    local_centroid: np.ndarray = field(init=False, default_factory=lambda: np.array([]))
    global_centroid: np.ndarray = field(
        init=False, default_factory=lambda: np.array([])
    )
    is_primary: bool = field(init=False, default=False)
    norm_local_coordinates: BoxCoordinates = field(
        init=False, default_factory=init_empty
    )

    @property
    def local_area(self):
        if self.local_coordinates:
            local_area = self.get_area(self.local_coordinates)
        else:
            raise AttributeError(
                "Local coordinates have to be defined for local area to be calculated."
            )
        return local_area

    # @property
    # def norm_local_area(self):
    #     if self.norm_local_coordinates:
    #         norm_local_area = self.get_area(self.norm_local_coordinates)
    #     else:
    #         raise AttributeError(
    #             "Normalized local coordinates have to be defined for local area to be calculated."
    #         )
    #     return norm_local_area

    @property
    def global_area(self):
        if self.global_coordinates:
            global_area = self.get_area(self.global_coordinates)
        else:
            raise AttributeError(
                "Global coordinates have to be defined for the global area to be calculated."
            )
        return global_area

    @property
    def config(self):
        _config = {
            "bbox_id": self.bbox_id,
            "image_id": self.image_id,
            "cutout_exists": self.cutout_exists,
            "is_primary": self.is_primary,
            "local_centroid":
            # list(self.local_centroid),
            list(self.norm_local_centroid),  # Always use normalized coordinates
            "local_coordinates":
            # self.local_coordinates.config,
            self.norm_local_coordinates.config,
            # Always use normalized coordinates
            "global_centroid": list(self.global_centroid),
            "global_coordinates": self.global_coordinates.config,
            "instance_rgb_id": self.instance_rgb_id,
            "cls": self.cls,
            "overlapping_bbox_ids": [box.bbox_id for box in self._overlapping_bboxes],
            "num_overlapping_bboxes": len(self._overlapping_bboxes),
            "non_target_weed": self.classifier_classname,
            "non_target_weed_pred_conf": self.classifier_confidence,
        }
        return _config

    def __post_init__(self):
        if not self.is_normalized:
            x, y = self.local_coordinates.top_left
            self.is_normalized = 0 <= x <= 1 and 0 <= y <= 1

        if self.local_coordinates:
            self.set_local_centroid()
            if self.is_normalized:
                self.norm_local_coordinates = self.local_coordinates.copy()
                self.norm_local_coordinates.is_scaleable = False
                self.set_norm_local_centroid()

        if self.global_coordinates:
            self.set_global_centroid()

        # A list of all overlapping bounding boxes
        self._overlapping_bboxes: List[BBox] = []

    def add_box(self, box):
        """Adds a box as an overlapping box

        Args:
            box (BBox): BBox to add as an overlapping box
        """
        self._overlapping_bboxes.append(box)

    def get_centroid(self, coords: BoxCoordinates) -> np.ndarray:
        """Get the centroid of the bounding box based on the coordinates passed

        Args:
            coords (BoxCoordinates): Bounding box coordinates

        Returns:
            np.ndarray: Centroid
        """
        centroid_x = (coords.bottom_right[0] + coords.bottom_left[0]) / 2.0
        centroid_y = (coords.bottom_left[1] + coords.top_left[1]) / 2.0
        centroid = np.array([centroid_x, centroid_y])
        return centroid

    def get_area(self, coordinates: BoxCoordinates) -> float:
        height = coordinates.bottom_left[1] - coordinates.top_left[1]
        width = coordinates.bottom_right[0] - coordinates.bottom_left[0]
        return float(height * width)

    def set_local_centroid(self):
        self.local_centroid = self.get_centroid(self.local_coordinates)

    def set_norm_local_centroid(self):
        self.norm_local_centroid = self.get_centroid(self.norm_local_coordinates)

    def set_global_centroid(self):
        self.global_centroid = self.get_centroid(self.global_coordinates)

    def set_local_scale(self, new_scale):
        self.local_coordinates.set_scale(new_scale)
        self.set_local_centroid()

    def update_global_coordinates(self, global_coordinates: BoxCoordinates):
        """Update the global coordinates of the bounding box

        Args:
            global_coordinates (BoxCoordinates): Global bounding box coordinates
        """
        assert not self.global_coordinates

        self.global_coordinates = global_coordinates
        self.global_centroid = self.get_centroid(self.global_coordinates)

    def bb_iou(self, comparison_box, type="global"):
        """Calculate the IoU of this bounding box with another bbox 'comparison_box',
        accounting for rotated bounding boxes by using all four corners.
        
        Args:
            comparison_box (BBox): Another bounding box
            type (str, optional): Use 'global' or 'local' coordinates. Defaults to "global".
        
        Returns:
            float: IoU of the two boxes
        """
        # Choose the coordinate set based on type
        if type == "global":
            _boxA = self.global_coordinates
            _boxB = comparison_box.global_coordinates
        elif type == "local":
            _boxA = self.local_coordinates
            _boxB = comparison_box.local_coordinates
        else:
            raise ValueError(f"Type {type} not supported.")

        # Create polygons from the four corners.
        # Make sure the corners are ordered consistently.
        polyA = Polygon([
            _boxA.top_left,
            _boxA.top_right,
            _boxA.bottom_right,
            _boxA.bottom_left
        ])
        polyB = Polygon([
            _boxB.top_left,
            _boxB.top_right,
            _boxB.bottom_right,
            _boxB.bottom_left
        ])

        # Optionally, clean up the geometries if they are invalid.
        if not polyA.is_valid:
            polyA = polyA.buffer(0)
        if not polyB.is_valid:
            polyB = polyB.buffer(0)

        # Check if the polygons intersect at all
        if not polyA.intersects(polyB):
            return 0.0
        
        try:
            inter_area = polyA.intersection(polyB).area
            union_area = polyA.union(polyB).area
        except Exception as e:
            # Log the error if desired
            log.error(f"Error calculating IoU: {e}")
            log.error(f"Box A: {polyA}")
            log.error(f"Box B: {polyB}")
            return 0.0
        
        # Avoid division by zero just in case
        if union_area == 0:
            return 0.0

        iou = inter_area / union_area
        return iou

    def assign_species(self, species):
        self.cls = species


# Batch Metadata ---------------------------------------------------------------------------


@dataclass
class BatchMetadata:
    """Batch metadata class for yaml loader"""

    # blob_home: str
    data_root: str
    batch_id: str
    upload_datetime: str
    image_list: List
    # schema_version: str = SCHEMA_VERSION


# Image dataclasses ----------------------------------------------------------


@dataclass
class ImageMetadata:
    ImageWidth: int
    ImageLength: int
    BitsPerSample: int
    Compression: int
    PhotometricInterpretation: int
    Make: str
    Model: str
    Orientation: int
    SamplesPerPixel: int
    XResolution: str
    YResolution: str
    PlanarConfiguration: int
    ResolutionUnit: int
    Software: str
    DateTime: str
    Rating: int
    ExifOffset: int
    ExposureTime: str
    FNumber: str
    ExposureProgram: int
    ISOSpeedRatings: int
    RecommendedExposureIndex: int
    ExifVersion: list
    DateTimeOriginal: str
    DateTimeDigitized: str
    BrightnessValue: str
    ExposureBiasValue: str
    MaxApertureValue: str
    MeteringMode: int
    LightSource: int
    Flash: int
    FocalLength: str
    FileSource: int
    SceneType: int
    CustomRendered: int
    ExposureMode: int
    WhiteBalance: int
    DigitalZoomRatio: str
    FocalLengthIn35mmFilm: int
    SceneCaptureType: int
    Contrast: int
    Saturation: int
    Sharpness: int
    LensModel: str
    LensSpecification: Optional[list] = None
    BodySerialNumber: Optional[str] = None
    MakerNote: Optional[str] = None
    ImageDescription: Optional[str] = None
    UserComment: Optional[str] = None
    ApplicationNotes: Optional[str] = None
    Tag: Optional[int] = None
    SubIFDs: Optional[int] = None


@dataclass
class CameraInfo:
    """ """

    camera_location: np.ndarray
    pixel_width: float
    pixel_height: float
    yaw: float
    pitch: float
    roll: float
    focal_length: float
    fov: BoxCoordinates.config = None


@dataclass
class Box:
    bbox_id: str
    image_id: str
    non_target_weed: str
    non_target_weed_pred_conf: float
    is_primary: Optional[bool]
    cutout_exists: Optional[bool]  # = field(default=False)
    local_centroid: list
    local_coordinates: BoxCoordinates
    global_centroid: list
    global_coordinates: BoxCoordinates
    cls: Optional[str]
    instance_rgb_id: Optional[List[int]]  # = field(default=None)
    overlapping_bbox_ids: List[BBox] = field(init=False, default_factory=lambda: [])

    def assign_species(self, species):
        self.cls = species


@dataclass
class BBoxFOV:
    top_left: list
    top_right: list
    bottom_left: list
    bottom_right: list


@dataclass
class BBoxMetadata:
    data_root: str
    batch_id: str
    image_path: str
    image_id: str
    width: int
    height: int
    camera_info: CameraInfo
    exif_meta: ImageMetadata
    bboxes: list[Box]


@dataclass
class Image:
    """Parent class for RemapImage and ImageData."""

    # blob_home: str
    data_root: str
    batch_id: str
    image_path: str
    image_id: str

    def __post_init__(self):
        image_array = self.array
        self.width = image_array.shape[1]
        self.height = image_array.shape[0]

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        img_path = Path(self.rel_path, self.image_id + ".jpg")
        img_array = cv2.imread(str(img_path))
        img_array = np.ascontiguousarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        return img_array

    @property
    def config(self):
        _config = {
            # "blob_home": self.blob_home,
            "data_root": self.data_root,
            "batch_id": self.batch_id,
            "image_id": self.image_id,
            "image_path": self.image_path,
            # "width": self.width,
            # "height": self.height,
            "exif_meta": asdict(self.exif_meta),
            "camera_info": asdict(self.camera_info),
            "bboxes": [box.config for box in self.bboxes],
            # "schema_version": self.schema_version
        }

        return _config

    def save_config(self, save_path):
        try:
            save_file = os.path.join(save_path, f"{self.image_id}.json")
            with open(save_file, "w") as f:
                json.dump(self.config, f, indent=4, default=str)
        except Exception as e:
            raise e
        return True


@dataclass
class RemapImage(Image):
    """For remapping labels (remap_labels)"""

    rel_path: str
    bboxes: list[BBox]
    camera_info: CameraInfo
    fullres_path: str
    width: int  # = field(init=False, default=-1)  #asfm scaled width
    height: int  # = field(init=False, default=-1)  #asfm scaled height
    exif_meta: Optional[ImageMetadata] = field(init=False, default=None)
    fullres_height: Optional[int] = field(init=False, default=-1)
    fullres_width: Optional[int] = field(init=False, default=-1)

    # schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        # self.height, self.width = self.array.shape[:2]
        # self.set_fullres_dims(self.width, self.height)
        self.exif_meta = self.get_exif()

    def set_fullres_dims(self, fullres_width, fullres_height):
        self.fullres_width = fullres_width
        self.fullres_height = fullres_height

    def get_exif(self):
        """Creates a dataclass by reading exif metadata, creating a dictionary, and creating dataclass form that dictionary"""
        # Open image file for reading (must be in binary mode)
        f = open(self.image_path, "rb")
        # Return Exif tags
        tags = exifread.process_file(f, details=False)
        f.close()
        meta = {}
        for x, y in tags.items():
            newval = (
                y.values[0]
                if type(y.values) == list and len(y.values) == 1
                else y.values
            )
            if type(newval) == exifread.utils.Ratio:
                newval = str(newval)
            meta[x.rsplit(" ")[1]] = newval
        imgmeta = ImageMetadata(**meta)
        return imgmeta

    @property
    def config(self):
        _config = super(RemapImage, self).config
        _config["fullres_width"] = self.fullres_width
        _config["fullres_height"] = self.fullres_height
        _config["rel_path"] = self.rel_path

        return _config


@dataclass
class ImageData(Image):
    """Dataclass for segmentation data generation"""

    rel_path: str
    width: Optional[int]
    height: Optional[int]
    exif_meta: ImageMetadata
    cutout_ids: List[str] = None
    camera_info: CameraInfo = None
    bboxes: list[Box] = None
    fullres_height: int = -1
    fullres_width: int = -1

    # schema_version: str = "1.0"

    def __post_init__(self):
        # Overload the post init of the super class
        # which reads the array for the width and height.
        # The width and height will be available in the metadata
        pass

    @property
    def config(self):
        _config = {
            # "blob_home": self.blob_home,
            "data_root": self.data_root,
            "batch_id": self.batch_id,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "fullres_height": self.fullres_height,
            "fullres_width": self.fullres_width,
            # "width": self.width,
            # "height": self.height,
            "exif_meta": asdict(self.exif_meta),
            "camera_info": asdict(self.camera_info),
            "bboxes": [asdict(x) for x in self.bboxes],
            # "bboxes": [x.config for x in self.bboxes],
            "cutout_ids": self.cutout_ids,
            "rel_path": self.rel_path,
            # "schema_version": self.schema_version
        }

        return _config

    def save_config(self, save_path):
        try:
            save_image_path = Path(save_path, self.image_id + ".json")
            with open(save_image_path, "w") as f:
                self.config.pop("rel_path")
                json.dump(self.config, f, indent=4, default=str)
        except Exception as e:
            raise e
        return True

    def save_mask(self, save_path, semantic_mask):
        fname = f"{self.image_id}.png"
        mask_path = Path(save_path, fname)
        cv2.imwrite(str(mask_path), cv2.cvtColor(semantic_mask, cv2.COLOR_RGB2BGR))
        return True


@dataclass
class Mask:
    mask_id: str
    mask_path: str
    image_id: str
    width: int = field(init=False, default=-1)
    height: int = field(init=False, default=-1)

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        mask_array = cv2.imread(self.mask_path)
        mask_array = np.ascontiguousarray(cv2.cvtColor(mask_array, cv2.COLOR_BGR2RGB))
        return mask_array

    def __post_init__(self):
        mask_array = self.array
        self.width = mask_array.shape[1]
        self.height = mask_array.shape[0]

    def save_mask(self, save_path):
        try:
            save_file = os.path.join(save_path, f"{self.image_id}.png")
            cv2.imwrite(save_file, self.array)

        except Exception as e:
            raise e
        return True


# Cutouts -------------------------------------------------------------------------------------


@dataclass
class CutoutProps:
    """Region properties for cutouts
    "area",  # float Area of the region i.e. number of pixels of the region scaled by pixel-area.
    "area_bbox",  # float Area of the bounding box i.e. number of pixels of bounding box scaled by pixel-area.
    "area_convex",  # float Are of the convex hull image, which is the smallest convex polygon that encloses the region.
    "axis_major_length",  # float The length of the major axis of the ellipse that has the same normalized second central moments as the region.
    "axis_minor_length",  # float The length of the minor axis of the ellipse that has the same normalized second central moments as the region.
    "centroid",  # array Centroid coordinate list [row, col].
    "eccentricity",  # float Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
    "extent",  # float Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)
    "solidity",  # float Ratio of pixels in the region to pixels of the convex hull image.
    "perimeter",  # float Perimeter of object which approximates the contour as a line
    "blur_effect", float, Compute a metric that indicates the strength of blur in an image (0 for no blur, 1 for maximal blur)
    "num_components", int number of connected mask components
    "color_distribution", dict (hex number, rgb, and occurnce) of top 12 most common colors. Excludes zero (black)
    "descriptive stats", dict, calculates descriptives stats of individual channels while excluding 0 (black)
    """

    area: Union[float, list]
    # area_bbox: Union[float, list]
    # area_convex: Union[float, list]
    # axis_major_length: Union[float, list]
    # axis_minor_length: Union[float, list]
    # centroid0: Union[float, list]
    # centroid1: Union[float, list]
    eccentricity: Union[float, list]
    # extent: float
    solidity: Union[float, list]
    perimeter: Union[float, list]
    # is_green: bool
    green_sum: int
    # exg_sum: float
    blur_effect: float
    num_components: int
    # color_distribution: dict
    cropout_rgb_mean: dict
    cropout_rgb_std: dict
    is_primary: bool = None
    extends_border: bool = None


# For Segmentation -------------------------------------------------------------------------------------


@dataclass
class Color:
    species: str
    hex: str = field(init=False)
    rgb: List[int] = field(init=False)

    def __post_init__(self):
        self.rgb = ""


@dataclass
class SegmentData:
    species_info: str
    species: str
    bbox: tuple


# Cutout -------------------------------------------------------------------------------------
@dataclass
class Cutout:
    """Per cutout. Goes to PlantCutouts"""

    # data_root: str
    season: str
    datetime: datetime.datetime  # Datetime of original image creation
    batch_id: str
    image_id: str
    cutout_id: str
    cutout_num: int
    cutout_props: CutoutProps
    cutout_height: int
    cutout_width: int
    category: dict
    # hwc: list
    # bbox: list
    cutout_path: str = None
    # exif_meta: dict = None
    # camera_info: dict = None
    # is_primary: bool = False
    # extends_border: bool = False
    # schema_version: str = SCHEMA_VERSION
    # synth: bool = False

    def __post_init__(self):
        self.cutout_num = int(self.cutout_id.split("_")[-1])
        self.cutout_path = str(Path(self.batch_id, self.cutout_id + ".png"))

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        cut_array = cv2.imread(self.cutout_path)
        cut_array = np.ascontiguousarray(cv2.cvtColor(cut_array, cv2.COLOR_BGR2RGB))
        return cut_array

    @property
    def config(self):
        _config = {
            # "data_root": self.data_root,
            "season": self.season,
            "datetime": self.datetime,
            "batch_id": self.batch_id,
            "image_id": self.image_id,
            "cutout_id": self.cutout_id,
            "cutout_num": self.cutout_num,
            "cutout_props": self.cutout_props,
            "cutout_height": self.cutout_height,
            "cutout_width": self.cutout_width,
            "category": self.category
            # "hwc": self.hwc,
            # "is_primary": self.is_primary,
            # "extends_border": self.extends_border,
            
            # "cutout_path": self.cutout_path,

            # "exif_meta": asdict(self.exif_meta),
            # "camera_info": asdict(self.camera_info),
            # "bbox": self.bbox,
            # "cls": self.cls,
            
        }

        return _config

    def save_config(self, save_dir):
        try:
            save_cutout_path = Path(save_dir, self.batch_id, self.cutout_id + ".json")
            with open(save_cutout_path, "w") as f:
                json.dump(self.config, f, indent=4, default=str)
        except Exception as e:
            raise e
        return True

    def save_cutout(self, save_dir, cutout_array):
        fname = f"{self.image_id}_{self.cutout_num}.png"
        cutout_path = Path(save_dir, self.batch_id, fname)
        cv2.imwrite(str(cutout_path), cv2.cvtColor(cutout_array, cv2.COLOR_RGB2BGRA))
        return True

    def save_cropout(self, save_dir, img_array):
        fname = f"{self.image_id}_{self.cutout_num}.jpg"
        cutout_path = Path(save_dir, self.batch_id, fname)
        cv2.imwrite(
            str(cutout_path),
            cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )

        return True

    def save_cutout_mask(self, save_dir, mask):
        fname = f"{self.image_id}_{self.cutout_num}_mask.png"
        mask_path = Path(save_dir, self.batch_id, fname)
        cv2.imwrite(str(mask_path), mask)
        return True


# Synthetic Data Generation -------------------------------------------------------------------------
@dataclass
class Pot:
    pot_path: str
    pot_id: uuid = None

    def __post_init__(self):
        self.pot_id = uuid.uuid4()

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        pot_array = cv2.imread(self.pot_path, cv2.IMREAD_UNCHANGED)
        pot_array = np.ascontiguousarray(cv2.cvtColor(pot_array, cv2.COLOR_BGR2RGBA))
        return pot_array

    @property
    def config(self):
        _config = {
            "pot_path": self.pot_path,
            "pot_id": self.pot_id,
        }
        return _config

    def save_config(self, save_path):
        try:
            save_image_path = Path(save_path, self.image_id + ".json")
            with open(save_image_path, "w") as f:
                json.dump(self.config, f, indent=4, default=str)
        except Exception as e:
            raise e
        return True


@dataclass
class Background:
    background_path: str
    background_id: uuid = None

    def __post_init__(self):
        self.background_id = uuid.uuid4()

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        background_array = cv2.imread(self.background_path)
        background_array = np.ascontiguousarray(
            cv2.cvtColor(background_array, cv2.COLOR_BGR2RGB)
        )
        return background_array

    @property
    def config(self):
        _config = {
            "background_path": self.background_path,
            "background_id": self.background_id,
        }
        return _config

    def save_config(self, save_path):
        try:
            save_image_path = Path(save_path, self.image_id + ".json")
            with open(save_image_path, "w") as f:
                json.dump(self.config, f, indent=4, default=str)
        except Exception as e:
            raise e
        return True


@dataclass
class SynthImage:
    data_root: str
    synth_path: str
    synth_maskpath: str
    pots: list[Pot]
    background: Background
    cutouts: list[Cutout]
    synth_id: str = field(init=False)

    def __post_init__(self):
        self.synth_id = uuid.uuid4()

    @property
    def config(self):
        _config = {
            "data_root": self.data_root,
            "background_id": self.background_id,
            "data_root": self.data_root,
            "synth_path": self.synth_path,
            "synth_maskpath": self.synth_maskpath,
            "pots": self.pots,
            "background": self.background,
            "cutouts": self.cutouts,
            "synth_id": self.synth_id,
        }
        return _config

    def save_config(self, save_path):
        try:
            save_image_path = Path(save_path, self.image_id + ".json")
            with open(save_image_path, "w") as f:
                json.dump(self.config, f, indent=4, default=str)
        except Exception as e:
            raise e
        return True


@dataclass
class SynthData:
    synthdir: str
    background_dir: str
    pot_dir: str
    cutout_dir: str
    cutout_csv: str
    cutouts: list[Cutout] = field(init=False, default=None)
    pots: list[Pot] = field(init=False, default=None)
    backgrounds: list[Background] = field(init=False, default=None)

    def __post_init__(self):
        self.backgrounds = self.get_backgrounds()
        self.pots = self.get_pots()
        self.cutouts = self.get_cutouts()

    def load_json(self, jsun):
        """Open json and create dictionary"""
        # Opening JSON file
        with open(jsun) as json_file:
            data = json.load(json_file)
        return data

    def get_pots(self):
        """Connnects documents in a database collection with items in a directory.
        Places connected items in a list of dataclasses.
        """
        docs = []
        meta_jsons = Path(self.pot_dir).glob("*.json")

        for meta in meta_jsons:
            meta_dict = self.load_json(meta)
            class_path = "pot" + "_path"
            # change path to suit local directory
            meta_dict[class_path] = str(
                Path(self.pot_dir) / Path(meta_dict[class_path]).name
            )
            dc = Pot(**meta_dict)
            docs.append(dc)
        return docs

    def get_backgrounds(self):
        docs = []
        meta_jsons = Path(self.background_dir).glob("*.json")

        for meta in meta_jsons:
            meta_dict = self.load_json(meta)
            class_path = "background" + "_path"
            # change path to suit local directory
            meta_dict[class_path] = str(
                Path(self.background_dir) / Path(meta_dict[class_path]).name
            )
            dc = Background(**meta_dict)
            docs.append(dc)
        return docs

    def get_cutouts(self):
        docs = []

        meta_jsons = Path(self.background_dir).glob("*.json")

        df = pd.read_csv(self.cutout_csv)
        df["temp_path"] = self.cutout_dir + "/" + df["cutout_path"]

        for _, meta in df.iterrows():
            meta_path = meta["temp_path"].replace(".png", ".json")
            meta_dict = self.load_json(meta_path)
            class_path = "cutout" + "_path"
            # change path to suit local directory
            meta_dict[class_path] = str(
                # Path(class_dir) / Path(meta_dict[class_path]).name)
                Path(self.cutout_dir)
                / Path(meta_dict[class_path]).name
            )

            dc = Cutout(**meta_dict)
            docs.append(dc)

        # print(docs)
        return docs


CUTOUT_PROPS = [
    "area",  # float Area of the region i.e. number of pixels of the region scaled by pixel-area.
    # "area_bbox",  # float Area of the bounding box i.e. number of pixels of bounding box scaled by pixel-area.
    # "area_convex",  # float Are of the convex hull image, which is the smallest convex polygon that encloses the region.
    # "axis_major_length",  # float The length of the major axis of the ellipse that has the same normalized second central moments as the region.
    # "axis_minor_length",  # float The length of the minor axis of the ellipse that has the same normalized second central moments as the region.
    # "centroid",  # array Centroid coordinate tuple (row, col).
    "eccentricity",  # float Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
    # "extent",  # float Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)
    "solidity",  # float Ratio of pixels in the region to pixels of the convex hull image.
    # "label",  # int The label in the labeled input image.
    "perimeter",  # float Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
    # "intensity_max",  # Float Value with the greatest intensity in the region.
    # "intensity_mean",  # flaot Value with the mean intensity in the region.
    # "intensity_min",  # float Value with the least intensity in the region.
    # "feret_diameter_maxfloat",  # flaot Maximum Feret’s diameter computed as the longest distance between points around a region’s convex hull contour as determined by find_contours
    # "equivalent_diameter_area",  # float The diameter of a circle with the same area as the region.
]
