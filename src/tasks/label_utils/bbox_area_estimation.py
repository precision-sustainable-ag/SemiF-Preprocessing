import math
from typing import List, Optional, Tuple


def sensor_pixel_pitch_mm(
    sensor_width_mm: float,
    sensor_height_mm: float,
    fullres_width_px: int,
    fullres_height_px: int,
) -> Tuple[float, float]:
    return sensor_width_mm / fullres_width_px, sensor_height_mm / fullres_height_px


def bbox_xywh_to_local_coordinates(
    bbox_xywh: Optional[List[int]],
    fullres_width_px: int,
    fullres_height_px: int,
) -> Optional[dict]:
    if not bbox_xywh or len(bbox_xywh) != 4:
        return None

    x, y, w, h = bbox_xywh
    if any(v is None for v in (x, y, w, h)):
        return None
    if fullres_width_px <= 0 or fullres_height_px <= 0:
        return None

    xmin = x / fullres_width_px
    ymin = y / fullres_height_px
    xmax = (x + w) / fullres_width_px
    ymax = (y + h) / fullres_height_px

    return {
        "top_left": [xmin, ymin],
        "top_right": [xmax, ymin],
        "bottom_left": [xmin, ymax],
        "bottom_right": [xmax, ymax],
        "local_centroid": [(xmin + xmax) / 2.0, (ymin + ymax) / 2.0],
        "is_normalized": True,
    }


def estimate_bbox_area_sqm(
    bbox_xywh: Optional[List[int]],
    pixel_width_mm: float,
    pixel_height_mm: float,
    focal_length_mm: float,
    z_axis_cm: float,
    cam_angle_deg: float,
) -> Optional[float]:
    """
    First-order ground-area estimate from image-space bbox dimensions using a
    pinhole-camera approximation and a fixed camera height.
    """
    if not bbox_xywh or len(bbox_xywh) != 4:
        return None

    _, _, bbox_w_px, bbox_h_px = bbox_xywh
    if any(v is None for v in (bbox_w_px, bbox_h_px)):
        return None
    if bbox_w_px <= 0 or bbox_h_px <= 0:
        return None
    if focal_length_mm <= 0 or z_axis_cm <= 0:
        return None

    z_axis_m = z_axis_cm / 100.0
    angle_rad = math.radians(cam_angle_deg or 0.0)
    cos_angle = math.cos(angle_rad)
    effective_distance_m = z_axis_m / cos_angle if cos_angle > 1e-6 else z_axis_m

    bbox_w_sensor_mm = bbox_w_px * pixel_width_mm
    bbox_h_sensor_mm = bbox_h_px * pixel_height_mm

    ground_w_m = effective_distance_m * (bbox_w_sensor_mm / focal_length_mm)
    ground_h_m = effective_distance_m * (bbox_h_sensor_mm / focal_length_mm)
    return float(ground_w_m * ground_h_m)
