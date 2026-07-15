#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import scipy.io as scio
from PIL import Image, ImageDraw
from graspnetAPI import GraspGroup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasps", default="outputs/latest/graspnet_target_grasps.npy")
    parser.add_argument("--meta", default="work/graspnet_input/meta.mat")
    parser.add_argument("--image", default="work/graspnet_input/color.png")
    parser.add_argument("--depth", default="work/graspnet_input/depth.png")
    parser.add_argument("--mask", default="outputs/latest/sam_mask_selected_raw.png")
    parser.add_argument("--summary", default="outputs/latest/yoloe_sam_selected_summary.json")
    parser.add_argument("--out-dir", default="outputs/latest")
    parser.add_argument("--min-bbox-y", type=float, default=0.30)
    parser.add_argument("--max-bbox-y", type=float, default=0.70)
    parser.add_argument("--min-mask-world-z-ratio", type=float, default=0.50)
    parser.add_argument(
        "--max-mapped-approach-y",
        type=float,
        default=0.20,
        help="Reject mapped TCP approach vectors that cut too strongly along world +Y.",
    )
    parser.add_argument(
        "--min-safe-score-ratio",
        type=float,
        default=0.45,
        help="Keep safe grasps whose score is at least this fraction of the best midbody score.",
    )
    parser.add_argument(
        "--max-front-approach-angle-deg",
        type=float,
        default=30.0,
        help="Maximum angle between GraspNet approach and camera-to-object direction.",
    )
    parser.add_argument("--pregrasp-distance", type=float, default=0.08)
    return parser.parse_args()


def normalize(vector):
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-9:
        return None
    return vector / norm


def vector_angle_deg(first, second):
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def project(k, xyz):
    uvw = k @ xyz
    if uvw[2] <= 1e-6:
        return None
    return np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]], dtype=float)


def mask_world_z_range(mask, depth_mm, k, camera_cv_to_world):
    ys, xs = np.where((mask > 0) & (depth_mm > 0))
    if len(xs) == 0:
        return None
    step = max(1, len(xs) // 8000)
    xs = xs[::step].astype(float)
    ys = ys[::step].astype(float)
    z = depth_mm[ys.astype(int), xs.astype(int)].astype(float) / 1000.0
    x = (xs - k[0, 2]) * z / k[0, 0]
    y = (ys - k[1, 2]) * z / k[1, 1]
    pts_cam = np.stack([x, y, z], axis=1)
    pts_world = pts_cam @ camera_cv_to_world[:3, :3].T + camera_cv_to_world[:3, 3]
    return float(np.percentile(pts_world[:, 2], 5)), float(np.percentile(pts_world[:, 2], 95))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = np.array(Image.open(args.image).convert("RGB"))
    h, w = image.shape[:2]
    mask = np.array(Image.open(args.mask).convert("L"))
    depth_mm = np.array(Image.open(args.depth))
    meta = scio.loadmat(args.meta)
    k = meta["intrinsic_matrix"].astype(float)
    camera_cv_to_world = meta["camera_cv_to_world"].astype(float)
    with open(args.summary, "r", encoding="utf-8") as f:
        selected = json.load(f)["selected"]
    x1, y1, x2, y2 = [float(v) for v in selected["bbox_xyxy"]]
    bbox_h = max(1.0, y2 - y1)
    allowed_y1 = y1 + args.min_bbox_y * bbox_h
    allowed_y2 = y1 + args.max_bbox_y * bbox_h

    z_range = mask_world_z_range(mask, depth_mm, k, camera_cv_to_world)
    if z_range is None:
        raise RuntimeError("mask has no valid depth pixels")
    mask_z_min, mask_z_max = z_range
    min_world_z = mask_z_min + args.min_mask_world_z_ratio * (mask_z_max - mask_z_min)

    gg = GraspGroup().from_npy(args.grasps)
    candidates = []
    rejected = {
        "outside_image": 0,
        "outside_mask": 0,
        "outside_mid_y": 0,
        "below_mid_z": 0,
        "unsafe_approach_y": 0,
        "not_front_approach": 0,
        "low_safe_score": 0,
    }
    for idx, grasp in enumerate(gg):
        t_cam = np.asarray(grasp.translation, dtype=float)
        uv = project(k, t_cam)
        if uv is None:
            rejected["outside_image"] += 1
            continue
        u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
        if u < 0 or u >= w or v < 0 or v >= h:
            rejected["outside_image"] += 1
            continue
        if mask[v, u] <= 0:
            rejected["outside_mask"] += 1
            continue
        if not (allowed_y1 <= v <= allowed_y2):
            rejected["outside_mid_y"] += 1
            continue
        r_world = camera_cv_to_world[:3, :3] @ np.asarray(grasp.rotation_matrix, dtype=float)
        t_world = camera_cv_to_world[:3, :3] @ t_cam + camera_cv_to_world[:3, 3]
        if float(t_world[2]) < min_world_z:
            rejected["below_mid_z"] += 1
            continue
        front_direction = normalize(t_world - camera_cv_to_world[:3, 3])
        if front_direction is None:
            rejected["not_front_approach"] += 1
            continue
        # GraspNet convention: rotation[:, 0] points from pregrasp toward the
        # object and rotation[:, 1] is the jaw opening axis.  The right-side
        # G1->TCP mapping used by IK maps these to TCP +Z and +X respectively.
        grasp_approach = normalize(r_world[:, 0])
        if grasp_approach is None:
            rejected["not_front_approach"] += 1
            continue
        front_angle_deg = vector_angle_deg(grasp_approach, front_direction)
        pregrasp_world = t_world - args.pregrasp_distance * grasp_approach
        robot_side = normalize(camera_cv_to_world[:3, 3] - t_world)
        pregrasp_robot_side_clearance = float(
            np.dot(pregrasp_world - t_world, robot_side)
        )
        if (
            front_angle_deg > args.max_front_approach_angle_deg
            or pregrasp_robot_side_clearance <= 0.0
        ):
            rejected["not_front_approach"] += 1
            continue
        candidates.append(
            {
                "idx": idx,
                "score": float(grasp.score),
                "uv": [float(uv[0]), float(uv[1])],
                "translation_camera": t_cam,
                "translation_world": t_world,
                "rotation_camera": np.asarray(grasp.rotation_matrix, dtype=float),
                "rotation_world": r_world,
                "width": float(grasp.width),
                "height": float(grasp.height),
                "depth": float(grasp.depth),
                "front_direction_world": front_direction,
                "front_approach_angle_deg": front_angle_deg,
                "pregrasp_world": pregrasp_world,
                "pregrasp_robot_side_clearance_m": pregrasp_robot_side_clearance,
            }
        )

    if not candidates:
        raise RuntimeError(
            "no grasp survived mid-body filters; "
            f"rejected={rejected}, allowed_y=({allowed_y1:.1f},{allowed_y2:.1f}), "
            f"min_world_z={min_world_z:.4f}, mask_z=({mask_z_min:.4f},{mask_z_max:.4f})"
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best_midbody_score = candidates[0]["score"]
    r_g1_to_g2 = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    for c in candidates:
        mapped_rotation = c["rotation_world"] @ r_g1_to_g2
        approach = mapped_rotation[:, 2]
        c["mapped_rotation_world"] = mapped_rotation
        c["mapped_approach_world"] = approach
        c["mapped_approach_y"] = float(approach[1])
        c["safe_score"] = float(
            c["score"]
            - 0.20 * max(0.0, c["mapped_approach_y"])
            - 0.004 * c["front_approach_angle_deg"]
        )

    safe_candidates = []
    for c in candidates:
        if c["score"] < args.min_safe_score_ratio * best_midbody_score:
            rejected["low_safe_score"] += 1
            continue
        if c["mapped_approach_y"] > args.max_mapped_approach_y:
            rejected["unsafe_approach_y"] += 1
            continue
        safe_candidates.append(c)

    selection_pool = safe_candidates if safe_candidates else candidates
    selection_pool.sort(key=lambda item: item["safe_score"], reverse=True)
    best = selection_pool[0]
    summary = {
        "selection": (
            "front-approach grasp after SAM mask + bbox middle-y + world-z "
            "midbody + camera-side pregrasp filters"
        ),
        "selected_grasp_index": best["idx"],
        "score": best["score"],
        "width": best["width"],
        "height": best["height"],
        "depth": best["depth"],
        "uv": best["uv"],
        "translation_camera": best["translation_camera"].tolist(),
        "rotation_camera": best["rotation_camera"].tolist(),
        "translation_world": best["translation_world"].tolist(),
        "rotation_world": best["rotation_world"].tolist(),
        "mapped_rotation_world": best["mapped_rotation_world"].tolist(),
        "front_direction_world": best["front_direction_world"].tolist(),
        "front_approach_angle_deg": best["front_approach_angle_deg"],
        "pregrasp_world": best["pregrasp_world"].tolist(),
        "pregrasp_robot_side_clearance_m": best[
            "pregrasp_robot_side_clearance_m"
        ],
        "camera_cv_to_world": camera_cv_to_world.tolist(),
        "filters": {
            "bbox_xyxy": [x1, y1, x2, y2],
            "allowed_y_px": [float(allowed_y1), float(allowed_y2)],
            "mask_world_z_percentile_5_95": [mask_z_min, mask_z_max],
            "min_world_z": min_world_z,
            "max_mapped_approach_y": args.max_mapped_approach_y,
            "min_safe_score_ratio": args.min_safe_score_ratio,
            "max_front_approach_angle_deg": args.max_front_approach_angle_deg,
            "pregrasp_distance_m": args.pregrasp_distance,
            "num_grasps": len(gg),
            "num_candidates": len(candidates),
            "num_safe_candidates": len(safe_candidates),
            "rejected": rejected,
        },
        "top_candidates": [
            {
                "idx": c["idx"],
                "score": c["score"],
                "uv": c["uv"],
                "translation_camera": c["translation_camera"].tolist(),
                "translation_world": c["translation_world"].tolist(),
                "rotation_camera": c["rotation_camera"].tolist(),
                "rotation_world": c["rotation_world"].tolist(),
                "mapped_rotation_world": c["mapped_rotation_world"].tolist(),
                "front_direction_world": c["front_direction_world"].tolist(),
                "front_approach_angle_deg": c["front_approach_angle_deg"],
                "pregrasp_world": c["pregrasp_world"].tolist(),
                "pregrasp_robot_side_clearance_m": c[
                    "pregrasp_robot_side_clearance_m"
                ],
                "mapped_approach_world": c["mapped_approach_world"].tolist(),
                "mapped_approach_y": c["mapped_approach_y"],
                "safe_score": c["safe_score"],
                "width": c["width"],
                "height": c["height"],
                "depth": c["depth"],
            }
            for c in candidates
        ],
    }
    json_path = out_dir / "graspnet_target_top_grasp_midbody.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    overlay = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((x1, allowed_y1, x2, allowed_y2), outline=(0, 255, 0), width=2)
    for c in candidates[:30]:
        u, v = c["uv"]
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(255, 180, 0))
    u, v = best["uv"]
    draw.ellipse((u - 9, v - 9, u + 9, v + 9), outline=(0, 255, 255), width=4)
    draw.line((u - 24, v, u + 24, v), fill=(0, 255, 255), width=3)
    draw.line((u, v - 24, u, v + 24), fill=(0, 255, 255), width=3)
    draw.text((10, 10), f"midbody GraspNet score {best['score']:.3f}", fill=(0, 255, 255))
    draw.text(
        (10, 28),
        f"safe candidates {len(safe_candidates)}/{len(candidates)}/{len(gg)}",
        fill=(0, 255, 255),
    )
    draw.text(
        (10, 46),
        f"approach_y {best['mapped_approach_y']:.3f}",
        fill=(0, 255, 255),
    )
    draw.text(
        (10, 64),
        f"front approach angle {best['front_approach_angle_deg']:.1f} deg",
        fill=(0, 255, 255),
    )
    overlay.save(out_dir / "graspnet_midbody_top_grasp_overlay.png")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
