#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import scipy.io as scio
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--detection", required=True)
    parser.add_argument("--grasp", required=True)
    parser.add_argument("--ik", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    return parser.parse_args()


def rotation_angle_deg(first, second):
    relative = np.asarray(first).T @ np.asarray(second)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def vector_angle_deg(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    cosine = float(
        np.clip(
            np.dot(first, second)
            / max(1.0e-12, np.linalg.norm(first) * np.linalg.norm(second)),
            -1.0,
            1.0,
        )
    )
    return math.degrees(math.acos(cosine))


def rgbd_points(image, depth_mm, intrinsic, camera_to_world, mask=None, stride=1):
    height, width = depth_mm.shape
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    depth = depth_mm[::stride, ::stride].astype(float) / 1000.0
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask[::stride, ::stride]
    z = depth[valid]
    px = xx[valid]
    py = yy[valid]
    camera = np.stack(
        [
            (px - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (py - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        ],
        axis=1,
    )
    world = camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    colors = image[::stride, ::stride][valid].astype(float) / 255.0
    return world, colors


def draw_frame(ax, origin, rotation, length, label_prefix, alpha=1.0, linewidth=2.4):
    colors = ("#e53935", "#43a047", "#1565c0")
    names = ("X/open", "Y/side", "Z/approach")
    for axis, (color, name) in enumerate(zip(colors, names)):
        vector = rotation[:, axis] * length
        ax.quiver(
            *origin,
            *vector,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            arrow_length_ratio=0.20,
            label=f"{label_prefix} {name}" if axis == 2 else None,
        )


def draw_gripper(ax, center, rotation, width, color, linestyle="-"):
    opening = rotation[:, 0]
    side = rotation[:, 1]
    approach = rotation[:, 2]
    half_width = float(np.clip(0.5 * width, 0.018, 0.045))
    palm = center - 0.040 * approach
    left = palm + half_width * opening
    right = palm - half_width * opening
    left_tip = left + 0.060 * approach
    right_tip = right + 0.060 * approach
    for start, end in ((left, right), (left, left_tip), (right, right_tip)):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linestyle=linestyle,
            linewidth=3.5,
        )
    side_a = palm - 0.015 * side
    side_b = palm + 0.015 * side
    ax.plot(
        [side_a[0], side_b[0]],
        [side_a[1], side_b[1]],
        [side_a[2], side_b[2]],
        color=color,
        linestyle=linestyle,
        linewidth=4.0,
    )


def setup_3d(ax, center, radius, title):
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_zlabel("world Z (m)")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-52)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.30)


def main():
    args = parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    detection = np.asarray(Image.open(args.detection).convert("RGB"))
    depth_mm = np.asarray(Image.open(args.depth))
    mask = np.asarray(Image.open(args.mask).convert("L")) > 0
    meta = scio.loadmat(args.meta)
    intrinsic = np.asarray(meta["intrinsic_matrix"], dtype=float)
    camera_to_world = np.asarray(meta["camera_cv_to_world"], dtype=float)
    with Path(args.grasp).open("r", encoding="utf-8") as handle:
        grasp = json.load(handle)
    with Path(args.ik).open("r", encoding="utf-8") as handle:
        ik = json.load(handle)

    scene_world, scene_colors = rgbd_points(
        image, depth_mm, intrinsic, camera_to_world, stride=3
    )
    object_world, object_colors = rgbd_points(
        image, depth_mm, intrinsic, camera_to_world, mask=mask, stride=1
    )
    desired = np.asarray(ik["target_world_xyz_m"], dtype=float)
    actual = np.asarray(ik["finger_center_world_xyz_m"], dtype=float)
    desired_rotation = np.asarray(ik["ik_target_rotation_world_3x3"], dtype=float)
    actual_rotation = np.asarray(ik["current_tool_rotation_world_3x3"], dtype=float)
    delta = actual - desired
    position_error = float(np.linalg.norm(delta))
    orientation_error = rotation_angle_deg(desired_rotation, actual_rotation)
    approach_error = vector_angle_deg(desired_rotation[:, 2], actual_rotation[:, 2])
    opening_error = vector_angle_deg(desired_rotation[:, 0], actual_rotation[:, 0])
    width = float(grasp.get("width", 0.06))

    metrics = {
        "target": "first-row AD Calcium Milk nearest the left gripper axis",
        "ik_converged": bool(ik["converged"]),
        "desired_tcp_world_m": desired.tolist(),
        "actual_tcp_world_m": actual.tolist(),
        "tcp_delta_actual_minus_desired_m": delta.tolist(),
        "tcp_position_error_m": position_error,
        "tcp_position_error_cm": position_error * 100.0,
        "tcp_orientation_error_deg": orientation_error,
        "approach_axis_error_deg": approach_error,
        "opening_axis_error_deg": opening_error,
        "desired_tcp_rotation_world_3x3": desired_rotation.tolist(),
        "actual_tcp_rotation_world_3x3": actual_rotation.tolist(),
        "graspnet_score": float(grasp["score"]),
        "grasp_pixel_uv": grasp["uv"],
        "object_point_count": int(len(object_world)),
    }
    Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(19, 7), dpi=150, facecolor="white")
    ax_rgb = fig.add_subplot(1, 3, 1)
    ax_rgb.imshow(detection)
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[..., 0] = 1.0
    overlay[..., 3] = mask.astype(float) * 0.28
    ax_rgb.imshow(overlay)
    bbox = grasp.get("filters", {}).get("bbox_xyxy")
    if bbox is not None:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        ax_rgb.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="#00e676",
                linewidth=2.5,
            )
        )
        ax_rgb.text(x1, max(5, y1 - 5), "first-row AD Calcium Milk", color="#00e676", fontsize=9)
    u, v = grasp["uv"]
    ax_rgb.scatter([u], [v], marker="x", s=140, linewidths=3, color="#00e5ff")
    ax_rgb.annotate(
        "GraspNet point",
        xy=(u, v),
        xytext=(u + 35, v - 22),
        color="#00e5ff",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#00e5ff", "lw": 1.8},
    )
    ax_rgb.set_title("Left wrist RGB | YOLOE box + SAM mask + grasp point")
    ax_rgb.axis("off")

    cloud_center = object_world.mean(axis=0)
    for panel, (radius, title) in enumerate(
        ((0.28, "RGB-D scene point cloud"), (0.13, "Target cloud and TCP alignment")),
        start=2,
    ):
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        if panel == 2:
            near = np.linalg.norm(scene_world - cloud_center, axis=1) < 0.48
            ax.scatter(
                scene_world[near, 0],
                scene_world[near, 1],
                scene_world[near, 2],
                c=scene_colors[near],
                s=0.8,
                alpha=0.25,
                depthshade=False,
            )
        ax.scatter(
            object_world[:, 0],
            object_world[:, 1],
            object_world[:, 2],
            c=object_colors,
            s=5.0 if panel == 3 else 3.0,
            alpha=1.0,
            depthshade=False,
            label="SAM target RGB-D cloud",
        )
        ax.scatter(*desired, marker="o", s=95, color="#00acc1", label="desired TCP")
        ax.scatter(*actual, marker="^", s=95, color="#8e24aa", label="IK actual TCP")
        ax.plot(
            [desired[0], actual[0]],
            [desired[1], actual[1]],
            [desired[2], actual[2]],
            linestyle=":",
            linewidth=3.0,
            color="#d81b60",
            label=f"position error {position_error * 100:.2f} cm",
        )
        draw_gripper(ax, desired, desired_rotation, width, "#00acc1", "--")
        draw_gripper(ax, actual, actual_rotation, width, "#8e24aa", "-")
        draw_frame(ax, desired, desired_rotation, 0.055, "desired", alpha=0.65)
        draw_frame(ax, actual, actual_rotation, 0.065, "actual", alpha=1.0)
        setup_3d(
            ax,
            cloud_center,
            radius,
            f"{title}\npos {position_error * 100:.2f} cm | rot {orientation_error:.2f} deg",
        )
        ax.legend(loc="upper left", fontsize=7)

    status = "PASS" if ik["converged"] else "NOT CONVERGED - preview only"
    fig.suptitle(
        "First-row AD Calcium Milk grasp preview | no arm command sent\n"
        f"IK {status} | delta XYZ = [{delta[0]*100:.2f}, {delta[1]*100:.2f}, {delta[2]*100:.2f}] cm | "
        f"approach {approach_error:.2f} deg | opening {opening_error:.2f} deg",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), w_pad=1.8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
