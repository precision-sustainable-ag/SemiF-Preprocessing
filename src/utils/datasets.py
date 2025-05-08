import logging
from typing import List, Dict

from dataclasses import dataclass, field

log = logging.getLogger(__name__)

@dataclass
class BBoxCoordinates:
    top_left: List[float]
    top_right: List[float]
    bottom_left: List[float]
    bottom_right: List[float]
    local_centroid: List[float]
    is_normalized: bool

@dataclass
class GlobalCoordinates:
    top_left: List[float]
    top_right: List[float]
    bottom_left: List[float]
    bottom_right: List[float]
    global_centroid: List[float]
    area_sqm: float

@dataclass
class FOV:
    height: List[float]
    width: List[float]
    top_left_xy: List[float]
    top_right_xy: List[float]
    bottom_left_xy: List[float]
    bottom_right_xy: bool
    fov_area_cm2: float
    
@dataclass
class CameraCoefficients:
    f: float
    cx: float
    cy: float
    b1: float
    b2: float
    k1: float
    k2: float
    k3: float
    k4: float
    p1: float
    p2: float

@dataclass
class CameraInfo:
    aligned: bool
    fov: FOV
    estimated_xyz: List[float]
    estimated_yaw: float
    estimated_pitch: float
    estimated_roll: float
    camera_coefficients: CameraCoefficients
    pixel_width: float
    pixel_height: float
    focal_length: float

@dataclass
class ExifMeta:
    ImageWidth: int
    ImageLength: int
    Make: str
    Model: str
    DateTime: str
    LensModel: str
    FocalLength: float
    FocalLengthIn35mmFilm: float

@dataclass
class BoundingBox:
    is_primary: bool
    cutout_exists: bool
    bbox_xywh: List[int] # Fullsized bbox coordinates in pixel coordinates
    category_class_id: int
    cutout_id: str
    non_target_weed: bool
    non_target_weed_pred_conf: float
    
    local_coordinates: BBoxCoordinates
    global_coordinates: GlobalCoordinates = None
    overlapping_cutout_ids: List[str] = field(default_factory=list)

@dataclass
class ImageMetadata:
    season: str
    datetime: str
    bbot_version: str
    batch_id: str
    image_id: str
    validated: bool
    version: str
    exif_meta: ExifMeta
    camera_info: CameraInfo
    fullres_height: int
    fullres_width: int
    downscaled_height: int
    downscaled_width: int
    
    annotations: List[BoundingBox] = field(default_factory=list)

    
