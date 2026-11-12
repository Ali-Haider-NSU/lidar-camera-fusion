"""Unit tests: run with `pytest` from the repository root."""
import numpy as np
import pytest

from src import scene
from src.pointcloud import crop, voxel_downsample, ransac_plane, euclidean_clusters, points_on_target
from src.projection import project_lidar_to_image
from src.kitti_io import (save_kitti_bin, load_kitti_bin, save_kitti_calib, load_kitti_calib,
                          project_kitti)


def test_points_on_target_child_at_20m():
    rows, cols, n = points_on_target(1.7, 0.5, 20)
    assert rows == pytest.approx(11.3, abs=0.1)
    assert cols == pytest.approx(7.2, abs=0.1)
    assert n == pytest.approx(81, abs=1)


def test_crop_keeps_only_roi():
    cloud = np.array([[5, 0, 0, 1], [60, 0, 0, 1], [5, 20, 0, 1], [5, 0, -5, 1]], float)
    out, mask = crop(cloud)
    assert mask.tolist() == [True, False, False, False]
    assert len(out) == 1


def test_voxel_downsample_centroids():
    cloud = np.array([[0.01, 0.01, 0.01, 1.0], [0.09, 0.05, 0.03, 3.0],   # same 0.2 m voxel
                      [1.0, 1.0, 1.0, 5.0]])                              # another voxel
    out, inv = voxel_downsample(cloud, 0.2)
    assert len(out) == 2
    first = out[inv[0]]
    assert first == pytest.approx([0.05, 0.03, 0.02, 2.0])


def test_ransac_finds_sloped_ground_and_ignores_wall():
    rng = np.random.default_rng(0)
    xy = rng.uniform([0, -10], [40, 10], (3000, 2))
    ground = np.c_[xy, -1.73 + 0.02 * xy[:, 0] + rng.normal(0, 0.02, 3000)]
    wall = np.c_[rng.uniform(0, 40, 3500), np.full(3500, 8.0), rng.uniform(-1.7, 3, 3500)]
    n, d, inl = ransac_plane(np.r_[ground, wall])          # the wall has MORE points
    assert -n[0] / n[2] == pytest.approx(0.02, abs=0.003)  # slope recovered
    assert inl[:3000].mean() > 0.99                         # ground found ...
    # the wall is not taken as ground; only its lowest ~15 cm lie inside the threshold
    wall_height_above_ground = wall[:, 2] - (-1.73 + 0.02 * wall[:, 0])
    assert not inl[3000:][wall_height_above_ground > 0.3].any()


def test_euclidean_clusters_separates_objects_and_drops_noise():
    rng = np.random.default_rng(1)
    a = rng.normal([0, 0, 0], 0.1, (50, 3))
    b = rng.normal([3, 0, 0], 0.1, (50, 3))
    noise = np.array([[10.0, 10.0, 0.0]])
    lab = euclidean_clusters(np.r_[a, b, noise], radius=0.6, min_size=8)
    assert len(set(lab[:50])) == 1 and len(set(lab[50:100])) == 1
    assert lab[0] != lab[50]
    assert lab[-1] == -2                                    # single point: noise


def test_projection_of_known_point():
    T = scene.camera_extrinsics()
    uv, depth, idx = project_lidar_to_image(np.array([[10.0, 1.0, 0.0]]), T, scene.K,
                                            scene.W_IMG, scene.H_IMG)
    assert uv[0] == pytest.approx([535.4, 167.0], abs=0.1)
    assert depth[0] == pytest.approx(9.73, abs=0.01)


def test_points_behind_camera_are_dropped():
    T = scene.camera_extrinsics()
    uv, depth, idx = project_lidar_to_image(np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), T,
                                            scene.K, scene.W_IMG, scene.H_IMG)
    assert idx.tolist() == [1]


def test_kitti_roundtrip_and_projection(tmp_path):
    rng = np.random.default_rng(2)
    cloud = np.c_[rng.uniform([5, -1, -1.5], [40, 1, 0.5], (500, 3)), rng.random(500)]
    cloud = cloud.astype(np.float32)
    T = scene.camera_extrinsics()
    save_kitti_bin(tmp_path / "000000.bin", cloud)
    save_kitti_calib(tmp_path / "000000.txt", scene.K, T)
    c2 = load_kitti_bin(tmp_path / "000000.bin")
    P2, R0, Tr = load_kitti_calib(tmp_path / "000000.txt")
    assert np.array_equal(c2, cloud)
    uv_k, _ = project_kitti(c2[:, :3], P2, R0, Tr)
    uv, _, _ = project_lidar_to_image(c2[:, :3], T, scene.K, 1e9, 1e9)   # no image bounds
    assert np.abs(uv_k - uv).max() < 0.01
