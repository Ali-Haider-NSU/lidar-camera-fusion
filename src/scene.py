"""Synthetic street scene with ground truth, a 64-beam LiDAR and a camera next to it.

Frames: LiDAR x forward, y left, z up (metres). Camera: optical frame (z forward, x right, y down).
Label ids: 1 ground, 2 wall, 3 pole, 10+ cars, 20+ pedestrians.
"""
import numpy as np

SENSOR_HEIGHT, SLOPE = 1.73, 0.02            # road rises 2 cm per metre ahead

BOXES = [  # (label, x_min, x_max, y_min, y_max, height)
    (10, 10.0, 14.5, -2.8, -1.0, 1.5), (11, 23.0, 27.5, 1.2, 3.0, 1.6),
    (12, 38.0, 42.6, -2.6, -0.7, 1.5), (2, 0.0, 60.0, 8.0, 8.4, 6.0)]
CYLINDERS = [  # (label, cx, cy, radius, height)
    (20, 9.0, 3.5, 0.3, 1.75), (21, 34.0, -4.5, 0.3, 1.75), (22, 17.0, 7.4, 0.3, 1.8),
    (3, 15.0, 4.6, 0.15, 4.0)]
NAMES = {10: "car A", 11: "car B", 12: "car C", 20: "pedestrian 1", 21: "pedestrian 2",
         22: "pedestrian 3", 3: "pole", 2: "wall"}
TRUE_DIST = {10: 10.0, 11: 23.0, 12: 38.0, 20: 8.7, 21: 33.7, 22: 16.7}   # closest surface (x)

# camera (geometry similar to KITTI's left colour camera)
W_IMG, H_IMG = 1242, 375
K = np.array([[721.5, 0, 609.6], [0, 721.5, 172.9], [0, 0, 1]])
R_CAM_LIDAR = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)   # body axes -> optical axes
CAM_POS = np.array([0.27, 0.0, -0.08])                               # camera in the LiDAR frame


def ground_z(x):
    return -SENSOR_HEIGHT + SLOPE * x


def ray_cast(origin, dirs, max_range=100.0):
    """First hit of every ray: returns distance t (inf = no hit) and label."""
    o = np.asarray(origin, float)
    t_best = np.full(len(dirs), np.inf)
    lab = np.zeros(len(dirs), int)
    den = dirs[:, 2] - SLOPE * dirs[:, 0]                 # ground plane z - SLOPE x = -H
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-SENSOR_HEIGHT - o[2] + SLOPE * o[0]) / den
    hit = (t > 0) & (t < t_best)
    t_best[hit], lab[hit] = t[hit], 1
    for label, x0, x1, y0, y1, h in BOXES:               # slab method for boxes
        z0 = ground_z((x0 + x1) / 2)
        lo, hi = np.array([x0, y0, z0]), np.array([x1, y1, z0 + h])
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (lo - o) / dirs, (hi - o) / dirs
        tn = np.nanmax(np.minimum(ta, tb), axis=1)
        tf = np.nanmin(np.maximum(ta, tb), axis=1)
        hit = (tn <= tf) & (tn > 0) & (tn < t_best)
        t_best[hit], lab[hit] = tn[hit], label
    for label, cx, cy, r, h in CYLINDERS:                # vertical cylinders
        dx, dy = dirs[:, 0], dirs[:, 1]
        ox, oy = o[0] - cx, o[1] - cy
        a = dx**2 + dy**2
        b = 2 * (ox * dx + oy * dy)
        c = ox**2 + oy**2 - r**2
        disc = b**2 - 4 * a * c
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (-b - np.sqrt(disc)) / (2 * a)
        z = o[2] + t * dirs[:, 2]
        zb = ground_z(cx)
        hit = (disc > 0) & (t > 0) & (z > zb) & (z < zb + h) & (t < t_best)
        t_best[hit], lab[hit] = t[hit], label
    t_best[t_best > max_range] = np.inf
    return t_best, lab


def simulate_scan(rng, n_beams=64, elev_range=(2.0, -24.8), h_step_deg=0.2,
                  noise=0.02, dropout=0.02):
    """One 360 deg scan of a rotating LiDAR. Returns points (N x 3), labels (N), cloud (N x 4)."""
    elev = np.radians(np.linspace(elev_range[0], elev_range[1], n_beams))
    azim = np.radians(np.arange(0, 360, h_step_deg))
    E, A = np.meshgrid(elev, azim, indexing="ij")
    dirs = np.c_[(np.cos(E) * np.cos(A)).ravel(), (np.cos(E) * np.sin(A)).ravel(),
                 np.sin(E).ravel()]
    rng_m, labels = ray_cast(np.zeros(3), dirs)
    ok = np.isfinite(rng_m) & (rng.random(len(rng_m)) > dropout)
    rng_m = rng_m + rng.normal(0, noise, len(rng_m))
    points = dirs[ok] * rng_m[ok, None]
    labels = labels[ok]
    intensity = np.clip(np.where(labels == 1, 0.2, 0.5) + rng.normal(0, 0.05, len(points)), 0, 1)
    cloud = np.c_[points, intensity].astype(np.float32)       # x, y, z, intensity
    return points, labels, cloud


def camera_extrinsics(R=R_CAM_LIDAR, cam_pos=CAM_POS):
    """4x4 T_cam_lidar: maps LiDAR points into the camera optical frame."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ cam_pos
    return T


def render_label_image(K=K, cam_pos=CAM_POS, R=R_CAM_LIDAR, width=W_IMG, height=H_IMG):
    """Ray-cast the scene from the camera: object label for every pixel."""
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    rays_cam = (np.linalg.inv(K) @ np.c_[uu.ravel(), vv.ravel(), np.ones(uu.size)].T).T
    rays_lidar = rays_cam @ R                                  # = (R^T d)^T
    rays_lidar /= np.linalg.norm(rays_lidar, axis=1, keepdims=True)
    _, lab_img = ray_cast(cam_pos, rays_lidar)
    return lab_img.reshape(height, width)


def object_corners(lab):
    """8 corners of a ground-truth object (box or cylinder) in the LiDAR frame."""
    for L, x0, x1, y0, y1, h in BOXES:
        if L == lab:
            z0 = ground_z((x0 + x1) / 2)
            return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z0 + h)])
    for L, cx, cy, r, h in CYLINDERS:
        if L == lab:
            z0 = ground_z(cx)
            return np.array([[cx + a, cy + b, z] for a in (-r, r) for b in (-r, r)
                             for z in (z0, z0 + h)])
    raise ValueError(f"unknown label {lab}")


def simulate_detections(label_image, T_cam_lidar, rng, K=K, labels=(10, 11, 12, 20, 21, 22)):
    """2D 'detector' boxes: projected 3D corners (whole object, also hidden parts) + 10 % margin
    + 2 px noise, for every object with at least 30 visible pixels."""
    detections = []
    for lab in labels:
        if np.sum(label_image == lab) < 30:
            continue
        c = (T_cam_lidar @ np.c_[object_corners(lab), np.ones(8)].T)[:3]
        uv_c = (K @ c)[:2] / (K @ c)[2]
        lo, hi = uv_c.min(1), uv_c.max(1)
        margin = 0.1 * (hi - lo)
        detections.append((lab, np.r_[lo - margin, hi + margin] + rng.normal(0, 2, 4)))
    return detections
