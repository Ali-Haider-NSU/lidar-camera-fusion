"""Classic LiDAR obstacle pipeline: crop, voxel grid, RANSAC ground plane, clustering, boxes."""
import numpy as np
from scipy.spatial import cKDTree


def points_on_target(height, width, distance, v_res_deg=0.43, h_res_deg=0.2):
    """Rough number of LiDAR points on an object: beams x columns that hit it (~ 1/d^2)."""
    rows = height / (distance * np.radians(v_res_deg))
    cols = width / (distance * np.radians(h_res_deg))
    return rows, cols, rows * cols


def crop(cloud, x=(0, 50), y=(-10, 10), z=(-3, 3)):
    """Keep points inside an axis-aligned region of interest; returns (cropped, mask)."""
    p = cloud[:, :3]
    m = ((p[:, 0] > x[0]) & (p[:, 0] < x[1]) & (p[:, 1] > y[0]) & (p[:, 1] < y[1])
         & (p[:, 2] > z[0]) & (p[:, 2] < z[1]))
    return cloud[m], m


def voxel_downsample(cloud, voxel=0.2):
    """Replace all points in each voxel by their centroid; returns (points, voxel index)."""
    idx = np.floor(cloud[:, :3] / voxel).astype(np.int64)
    _, inverse, counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    out = np.zeros((len(counts), cloud.shape[1]))
    np.add.at(out, inverse, cloud)
    return out / counts[:, None], inverse


def ransac_plane(p, threshold=0.15, iterations=200, seed=0, min_abs_nz=0.9):
    """Fit the dominant roughly horizontal plane n.p + d = 0 with RANSAC, then refit.

    Returns (n, d, inlier mask). min_abs_nz rejects walls (normal must be near vertical).
    """
    r = np.random.default_rng(seed)
    best = np.zeros(len(p), bool)
    for _ in range(iterations):
        a, b, c = p[r.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < min_abs_nz:
            continue
        inl = np.abs((p - a) @ n) < threshold
        if inl.sum() > best.sum():
            best = inl
    q = p[best]                                   # least-squares refit on the inliers
    c = q.mean(0)
    n = np.linalg.svd(q - c, full_matrices=False)[2][-1]
    return n, -n @ c, np.abs((p - c) @ n) < threshold


def euclidean_clusters(pts, radius=0.6, min_size=8):
    """Region growing with a KD-tree. Labels: cluster id >= 0, -2 for clusters below min_size."""
    tree = cKDTree(pts)
    lab = np.full(len(pts), -1)
    cid = 0
    for seed in range(len(pts)):
        if lab[seed] != -1:
            continue
        queue, members = [seed], []
        lab[seed] = cid
        while queue:
            i = queue.pop()
            members.append(i)
            for j in tree.query_ball_point(pts[i], radius):
                if lab[j] == -1:
                    lab[j] = cid
                    queue.append(j)
        if len(members) < min_size:
            lab[members] = -2
        else:
            cid += 1
    return lab


def cluster_box(pts):
    """Axis-aligned box (min, max corner) and the horizontal distance to the closest point."""
    lo, hi = pts.min(0), pts.max(0)
    closest = np.min(np.linalg.norm(pts[:, :2], axis=1))
    return lo, hi, closest
