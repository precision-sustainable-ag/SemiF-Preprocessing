import logging
from typing import List, Dict, Optional

from dataclasses import dataclass, field

log = logging.getLogger(__name__)

@dataclass
class BBoxCoordinates:
    top_left: List[float] = None
    top_right: List[float] = None
    bottom_left: List[float] = None
    bottom_right: List[float] = None
    local_centroid: List[float] = None
    is_normalized: Optional[bool] = None

@dataclass
class GlobalCoordinates:
    top_left: Optional[List[float]] = None
    top_right: Optional[List[float]] = None
    bottom_left: Optional[List[float]] = None
    bottom_right: Optional[List[float]] = None
    global_centroid: Optional[List[float]] = None
    area_sqm: Optional[float] = None

@dataclass
class FOV:
    height: float = None
    width: float = None
    top_left_xy: List[float] = None
    top_right_xy: List[float] = None
    bottom_left_xy: List[float] = None
    bottom_right_xy: List[float] = None
    fov_area_cm2: float = None

@dataclass
class CameraCoefficients:
    f: float = None
    cx: float = None
    cy: float = None
    b1: float = None
    b2: float = None
    k1: float = None
    k2: float = None
    k3: float = None
    k4: float = None
    p1: float = None
    p2: float = None

@dataclass
class CameraInfo:
    fov: FOV
    camera_coefficients: CameraCoefficients
    aligned: bool = None
    estimated_xyz: List[float] = None
    estimated_yaw: float = None
    estimated_pitch: float = None
    estimated_roll: float = None
    pixel_width: float = None
    pixel_height: float = None
    focal_length: float = None

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
    cutout_id: str
    category_class_id: int
    is_primary: bool = None
    cutout_exists: bool = None
    bbox_xywh: List[int] = None
    non_target_weed: bool = None
    non_target_weed_pred_conf: float = None
    local_coordinates: BBoxCoordinates = None
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
    downscaled_height: int = None
    downscaled_width: int = None
    annotations: List[BoundingBox] = field(default_factory=list)

    
