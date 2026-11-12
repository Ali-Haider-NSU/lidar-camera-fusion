"""Distance for a camera 2D box from the LiDAR points projected into it."""
import numpy as np

from .projection import project_lidar_to_image

METHODS = ["median", "nearest group", "largest group", "best cluster"]


def box_distance(box, method, points, T_cam_lidar, K, width, height, ground_mask, cluster_id,
                 cam_x=0.0, gap=1.0):
    """Distance (along LiDAR x) of the object in a 2D box x1 y1 x2 y2.

    median        : median depth of all points in the box
    nearest group : split ground-free depths at jumps > gap, take the nearest group
    largest group : ... take the group with the most points
    best cluster  : the 3D cluster whose projection lies most completely inside the box
    cam_x converts camera depth into LiDAR x (camera mounted cam_x ahead of the LiDAR).
    """
    uv, depth, idx = project_lidar_to_image(points, T_cam_lidar, K, width, height)
    u, v = uv[:, 0], uv[:, 1]
    inb = (u >= box[0]) & (u <= box[2]) & (v >= box[1]) & (v <= box[3])
    if method == "median":
        z = depth[inb]
        return np.median(z) + cam_x if len(z) else np.nan
    z = np.sort(depth[inb & ~ground_mask[idx]])
    if len(z) == 0:
        return np.nan
    groups = np.split(z, np.nonzero(np.diff(z) > gap)[0] + 1)
    if method == "nearest group":
        return np.median(groups[0]) + cam_x
    if method == "largest group":
        return np.median(max(groups, key=len)) + cam_x
    if method == "best cluster":
        best, best_score = None, (0, 0)
        for c in np.unique(cluster_id[idx[inb]]):
            if c < 0:
                continue
            inside = np.sum(cluster_id[idx[inb]] == c)
            score = (inside / np.sum(cluster_id[idx] == c), inside)
            if score > best_score:
                best, best_score = c, score
        return points[cluster_id == best, 0].min() if best is not None else np.nan
    raise ValueError(f"unknown method {method}")
