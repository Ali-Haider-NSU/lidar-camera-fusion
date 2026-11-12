"""Full pipeline on the synthetic street: LiDAR scan -> ground removal -> clustering -> projection
into the camera -> distance for every camera detection -> effect of calibration error.

Run from the repository root:   python -m examples.run_pipeline [--figures]
"""
import sys

import numpy as np
from sklearn.cluster import DBSCAN

from src import scene
from src.pointcloud import (crop, voxel_downsample, ransac_plane, euclidean_clusters,
                            cluster_box, points_on_target)
from src.projection import project_lidar_to_image, rotz
from src.fusion import box_distance, METHODS


def main(make_figures=False):
    rng = np.random.default_rng(0)

    # ---------- 1. LiDAR scan ----------
    points, labels, cloud = scene.simulate_scan(rng)
    print(f"LiDAR scan: {len(cloud)} points")
    for name, lab in [("pedestrian at  9 m", 20), ("pedestrian at 34 m", 21)]:
        expected = points_on_target(1.75, 0.6, 9 if lab == 20 else 34)[2]
        print(f"  {name}: {np.sum(labels == lab):4d} points (rule of thumb ~{expected:.0f})")

    # ---------- 2. crop, voxel grid, ground removal ----------
    roi, roi_mask = crop(cloud)
    roi_labels = labels[roi_mask]
    vox, _ = voxel_downsample(roi, 0.2)
    print(f"\nregion of interest {len(roi)} points, after 0.2 m voxel grid {len(vox)} "
          f"({len(vox) / len(roi):.0%})")

    p = roi[:, :3]
    is_ground = roi_labels == 1
    n, d, ground_ransac = ransac_plane(p)
    ground_naive = p[:, 2] < -scene.SENSOR_HEIGHT + 0.2
    print("\n| Ground removal | Recall | Precision | Obstacle points removed |")
    print("|---|---|---|---|")
    for name, pred in [("height threshold", ground_naive), ("RANSAC plane", ground_ransac)]:
        tp = np.sum(pred & is_ground)
        print(f"| {name} | {tp / is_ground.sum():.1%} | {tp / pred.sum():.1%} | "
              f"{np.sum(pred & ~is_ground)} |")
    print(f"fitted slope {-n[0] / n[2]:.3f} (true {scene.SLOPE})")

    # ---------- 3. clustering ----------
    obst = p[~ground_ransac]
    obst_true = roi_labels[~ground_ransac]
    lab_kd = euclidean_clusters(obst)
    lab_db = DBSCAN(eps=0.6, min_samples=5).fit_predict(obst)
    objects = {l for l in np.unique(obst_true) if l >= 2}
    for name, lab in [("KD-tree clustering", lab_kd), ("DBSCAN", lab_db)]:
        ids = [c for c in np.unique(lab) if c >= 0]
        found = {np.bincount(obst_true[lab == c]).argmax() for c in ids}
        print(f"{name}: {len(ids)} clusters, objects found {len(found & objects)} of {len(objects)}")

    rows = []
    for c in [c for c in np.unique(lab_kd) if c >= 0]:
        pts = obst[lab_kd == c]
        if len(pts) >= 30:
            lo, hi, closest = cluster_box(pts)
            true = np.bincount(obst_true[lab_kd == c]).argmax()
            rows.append((closest, scene.NAMES.get(true, true), len(pts), hi - lo))
    print("\nclusters with >= 30 points:")
    for closest, name, npts, size in sorted(rows):
        print(f"  {name:13s} {npts:5d} pts  box {size[0]:4.1f} x {size[1]:4.1f} x {size[2]:3.1f} m"
              f"  closest {closest:5.1f} m")

    # ---------- 4. camera and projection ----------
    T = scene.camera_extrinsics()
    label_image = scene.render_label_image()
    detections = scene.simulate_detections(label_image, T, rng)
    uv, depth, idx = project_lidar_to_image(points, T, scene.K, scene.W_IMG, scene.H_IMG)
    hit = label_image[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    print(f"\n{len(uv)} of {len(points)} points land in the image; "
          f"{np.mean(hit == labels[idx]):.1%} on the same object")

    # ---------- 5. distance for each camera detection ----------
    ground_all = np.abs(points @ n + d) < 0.15
    cluster_id = np.full(len(points), -1)
    cluster_id[np.nonzero(roi_mask)[0][~ground_ransac]] = lab_kd
    kw = dict(points=points, T_cam_lidar=T, K=scene.K, width=scene.W_IMG, height=scene.H_IMG,
              ground_mask=ground_all, cluster_id=cluster_id, cam_x=scene.CAM_POS[0])
    print("\n| Object | True (m) | " + " | ".join(METHODS) + " |")
    print("|---|---|" + "---|" * len(METHODS))
    for lab, box in detections:
        est = [box_distance(box, m, **kw) for m in METHODS]
        print(f"| {scene.NAMES[lab]} | {scene.TRUE_DIST[lab]:.1f} | "
              + " | ".join(f"{e:.2f}" for e in est) + " |")

    # ---------- 6. calibration error ----------
    def on_correct_object(T_, lab):
        uv_, _, idx_ = project_lidar_to_image(points, T_, scene.K, scene.W_IMG, scene.H_IMG)
        sel = labels[idx_] == lab
        seen = label_image[uv_[sel, 1].astype(int), uv_[sel, 0].astype(int)]
        return np.mean(seen == lab)

    print("\n| Yaw error | Shift | car 10 m | car 38 m | ped 9 m | ped 17 m |")
    print("|---|---|---|---|---|---|")
    curves = {}
    for err in [0.0, 0.25, 0.5, 1.0, 2.0]:
        T_bad = T.copy()
        T_bad[:3, :3] = scene.R_CAM_LIDAR @ rotz(err)
        share = [on_correct_object(T_bad, lab) for lab in (10, 12, 20, 22)]
        curves[err] = share
        print(f"| {err:.2f}° | {721.5 * np.tan(np.radians(err)):.1f} px | "
              + " | ".join(f"{s:.1%}" for s in share) + " |")

    if make_figures:
        save_figures(points, labels, obst, lab_kd, uv, depth, label_image, detections, curves)


def save_figures(points, labels, obst, lab_kd, uv, depth, label_image, detections, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.scatter(points[:, 0], points[:, 1], s=0.1, c="lightgrey")
    keep = lab_kd >= 0
    ax.scatter(obst[keep, 0], obst[keep, 1], s=0.6, c=lab_kd[keep] % 20, cmap="tab20")
    ax.set_xlim(0, 50); ax.set_ylim(-10, 10); ax.set_aspect("equal")
    ax.set_xlabel("x forward (m)"); ax.set_ylabel("y left (m)")
    ax.set_title("Bird's-eye view: all points (grey), obstacle clusters (colours)")
    fig.tight_layout(); fig.savefig("docs/images/bev_clusters.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.imshow(label_image, cmap="tab20", alpha=0.25)
    sc = ax.scatter(uv[:, 0], uv[:, 1], s=1, c=depth, cmap="turbo", vmax=45)
    for lab, b in detections:
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                   ec="black", lw=1))
    ax.set_xlim(0, scene.W_IMG); ax.set_ylim(scene.H_IMG, 0)
    fig.colorbar(sc, label="depth (m)", fraction=0.02)
    ax.set_title("LiDAR points projected into the camera image, with detector boxes")
    fig.tight_layout(); fig.savefig("docs/images/projection.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    errs = sorted(curves)
    for k, name in enumerate(["car 10 m", "car 38 m", "pedestrian 9 m", "pedestrian 17 m"]):
        ax.plot(errs, [100 * curves[e][k] for e in errs], "o-", label=name)
    ax.set_xlabel("extrinsic yaw error (deg)"); ax.set_ylabel("points on the right object (%)")
    ax.set_title("Effect of camera-LiDAR calibration error"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("docs/images/calibration_error.png", dpi=130); plt.close(fig)
    print("\nfigures written to docs/images/")


if __name__ == "__main__":
    main(make_figures="--figures" in sys.argv)
