"""Projection of LiDAR points into a camera image."""
import numpy as np


def rotz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def project_lidar_to_image(pts_lidar, T_cam_lidar, K, width, height, min_depth=0.5):
    """LiDAR points -> pixels and camera depth, for points in front of the camera and inside
    the image. Returns (uv, depth, idx) with idx = index of each kept point in pts_lidar."""
    p_h = np.c_[pts_lidar, np.ones(len(pts_lidar))]
    p_cam = (T_cam_lidar @ p_h.T).T[:, :3]                  # 1. into the camera frame
    front = p_cam[:, 2] > min_depth                         # 2. drop points behind the camera
    uvw = (K @ p_cam[front].T).T                            # 3. pinhole projection
    uv = uvw[:, :2] / uvw[:, 2:]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    idx = np.nonzero(front)[0][inside]
    return uv[inside], p_cam[front][inside, 2], idx
