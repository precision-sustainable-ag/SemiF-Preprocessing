import pandas as pd
import numpy as np
from geopy.distance import geodesic
import cv2

def rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """Constructs a 3D rotation matrix from yaw, pitch, roll in degrees."""
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0,              1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0,           0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    return Rz @ Ry @ Rx

def pixel_to_world_with_manual_b1b2(camera_csv_path, image_label, u, v):
    """
    Fully manual version of pixel to world conversion using b1, b2, and full distortion correction.
    """
    df = pd.read_csv(camera_csv_path)
    cam = df[df['label'] == image_label].iloc[0]

    K = np.array([
        [cam['f'], cam['b2'], cam['cx']],
        [0, cam['f']*(1+cam['b1']), cam['cy']],
        [0, 0, 1]
    ])

    dist = np.array([cam['k1'], cam['k2'], cam['p1'], cam['p2'], cam['k3']])

    x_u, y_u = cv2.undistortPoints(
        np.array([[[u, v]]], dtype=np.float32), K, dist).squeeze()

    ray_cam = np.array([x_u, y_u, 1.0])
    
    R = rotation_matrix(cam['Estimated_Yaw'], cam['Estimated_Pitch'], cam['Estimated_Roll'])
    ray_world = R @ ray_cam
    cam_center = np.array([cam['Estimated_X'], cam['Estimated_Y'], cam['Estimated_Z']])
    scale = 0 / ray_world[2]
    # scale = (cam_center[2] - cam_center[2]) / ray_world[2]
    point_world = cam_center + scale * ray_world

    return point_world

if __name__ == "__main__":
    # marker 424
    csv_path = "data/longterm_images2/semifield-developed-images/NC_2025-04-22/autosfm2C/reference/camera_reference.csv"
    image_name = "NC_1745339367"
    pixel = (8843, 4435)
    actual = (35.77418278, -78.67260404) # 424

    # # marker 488
    # csv_path = "data/longterm_images2/semifield-developed-images/NC_2025-04-22/autosfm1A/reference/camera_reference.csv"
    # image_name = "NC_1745334114"
    # pixel = (6743, 4139)  
    # actual = (35.77417572, -78.67259151)  # 488


    world_xyz = pixel_to_world_with_manual_b1b2(
        camera_csv_path=csv_path,
        image_label=image_name,
        u=pixel[0],
        v=pixel[1])

    # calculated (lat, lon)
    calc = (world_xyz[1], world_xyz[0])

    # Calculate error/distance in meters
    error_meters = geodesic(actual, calc).meters * 100
    print(f"Geodesic error: {error_meters:.3f} centimeters")