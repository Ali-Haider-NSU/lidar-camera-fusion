"""Read and write point clouds and calibration in the KITTI object-detection format."""
import numpy as np


def save_kitti_bin(path, cloud):
    np.asarray(cloud, dtype=np.float32).tofile(path)       # x, y, z, reflectance as float32


def load_kitti_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def save_kitti_calib(path, K, T_cam_lidar):
    P2 = np.c_[K, np.zeros(3)]
    with open(path, "w") as f:
        f.write("P2: " + " ".join(f"{v:.6e}" for v in P2.ravel()) + "\n")
        f.write("R0_rect: " + " ".join(f"{v:.6e}" for v in np.eye(3).ravel()) + "\n")
        f.write("Tr_velo_to_cam: " + " ".join(f"{v:.6e}" for v in T_cam_lidar[:3].ravel()) + "\n")


def load_kitti_calib(path):
    """Return P2 (3x4), R0_rect (4x4) and Tr_velo_to_cam (4x4) from a KITTI calib file."""
    vals = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                key, rest = line.split(":", 1)
                vals[key.strip()] = np.array([float(x) for x in rest.split()])
    P2 = vals["P2"].reshape(3, 4)
    R0 = np.eye(4)
    R0[:3, :3] = vals["R0_rect"].reshape(3, 3)
    Tr = np.eye(4)
    Tr[:3] = vals["Tr_velo_to_cam"].reshape(3, 4)
    return P2, R0, Tr


def project_kitti(pts, P2, R0, Tr, min_depth=0.5):
    """KITTI convention: x_img = P2 * R0_rect * Tr_velo_to_cam * X_velo. Returns (uv, depth)."""
    X = np.c_[pts, np.ones(len(pts))].T
    cam = R0 @ Tr @ X
    front = cam[2] > min_depth
    uvw = P2 @ cam[:, front]
    return (uvw[:2] / uvw[2]).T, cam[2, front]
