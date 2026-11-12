"""The same obstacle pipeline with Open3D (reference sketch, not run in this repository).

pip install open3d
python -m extras.open3d_pipeline                      # simulated scan
python -m extras.open3d_pipeline path/to/000000.bin   # a KITTI scan
"""
import sys
import numpy as np
import open3d as o3d

from src import scene
from src.kitti_io import load_kitti_bin

if len(sys.argv) > 1:
    cloud = load_kitti_bin(sys.argv[1])
else:
    _, _, cloud = scene.simulate_scan(np.random.default_rng(0))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cloud[:, :3].astype(np.float64))

roi_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([0.0, -10.0, -3.0]),
                                              max_bound=np.array([50.0, 10.0, 3.0]))
roi = pcd.crop(roi_box)
down = roi.voxel_down_sample(voxel_size=0.2)

plane, inliers = down.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=200)
ground = down.select_by_index(inliers)
obstacles = down.select_by_index(inliers, invert=True)

labels = np.array(obstacles.cluster_dbscan(eps=0.6, min_points=5))
boxes = []
for c in range(labels.max() + 1):
    cluster = obstacles.select_by_index(np.nonzero(labels == c)[0])
    box = cluster.get_axis_aligned_bounding_box()
    box.color = (1, 0, 0)
    boxes.append(box)

ground.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([ground, obstacles, *boxes])      # opens a 3D viewer window
