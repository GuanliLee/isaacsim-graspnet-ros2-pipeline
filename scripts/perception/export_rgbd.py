#!/usr/bin/env python3
import argparse
import asyncio
import os
import time

from isaacsim import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--usd",
        default="/workspace/assets/scene.usd",
    )
    parser.add_argument(
        "--camera",
        default="/World/Robot/lifting_link/head_view_camera",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/grasp-pipeline/work/capture",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup-frames", type=int, default=80)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--robot-base", nargs=4, type=float, metavar=("X", "Y", "Z", "YAW"))
    parser.add_argument("--lift", type=float)
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import omni
import omni.replicator.core as rep
import scipy.io as scio
from PIL import Image
from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.sensors.camera import Camera
from pxr import Gf, Usd, UsdGeom


def camera_intrinsic_matrix(camera_prim, width, height):
    usd_cam = UsdGeom.Camera(camera_prim)
    focal_length = float(usd_cam.GetFocalLengthAttr().Get())
    horiz_aperture = float(usd_cam.GetHorizontalApertureAttr().Get())
    vert_aperture_attr = usd_cam.GetVerticalApertureAttr().Get()
    if vert_aperture_attr is None or float(vert_aperture_attr) == 0.0:
        vert_aperture = horiz_aperture * float(height) / float(width)
    else:
        vert_aperture = float(vert_aperture_attr)

    fx = width * focal_length / horiz_aperture
    fy = height * focal_length / vert_aperture
    cx = width * 0.5
    cy = height * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def camera_cv_to_world_matrix(camera_prim):
    local_to_world = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    origin = local_to_world.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    usd_x = local_to_world.Transform(Gf.Vec3d(1.0, 0.0, 0.0)) - origin
    usd_y = local_to_world.Transform(Gf.Vec3d(0.0, 1.0, 0.0)) - origin
    usd_z = local_to_world.Transform(Gf.Vec3d(0.0, 0.0, 1.0)) - origin

    # GraspNet point clouds use the common CV camera frame:
    # +X right, +Y down, +Z forward. USD cameras use +X right, +Y up, -Z forward.
    cv_x_world = np.array(usd_x, dtype=np.float32)
    cv_y_world = -np.array(usd_y, dtype=np.float32)
    cv_z_world = -np.array(usd_z, dtype=np.float32)
    cv_origin_world = np.array(origin, dtype=np.float32)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, 0] = cv_x_world
    transform[:3, 1] = cv_y_world
    transform[:3, 2] = cv_z_world
    transform[:3, 3] = cv_origin_world
    return transform


def get_frame_value(frame, *names):
    for name in names:
        if name in frame:
            value = frame[name]
            if isinstance(value, dict) and "data" in value:
                value = value["data"]
            return value
    raise KeyError(f"none of {names} are present; available keys={list(frame.keys())}")


def set_robot_pose(stage, x, y, z, yaw):
    robot = stage.GetPrimAtPath("/World/Robot")
    if not robot.IsValid():
        raise RuntimeError("robot prim not found: /World/Robot")
    xform = UsdGeom.Xformable(robot)
    ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
    (ops.get("xformOp:translate") or xform.AddTranslateOp()).Set(Gf.Vec3d(x, y, z))
    (ops.get("xformOp:rotateZ") or xform.AddRotateZOp()).Set(np.degrees(yaw))


def set_lift(stage, lift):
    joint = stage.GetPrimAtPath("/World/Robot/joints/lifting_joint")
    if not joint.IsValid():
        return
    for attr_name in ("drive:linear:physics:targetPosition", "state:linear:physics:position"):
        attr = joint.GetAttribute(attr_name)
        if attr and attr.IsValid():
            attr.Set(float(lift))


async def capture_replicator_rgbd(camera_path, width, height, warmup_frames):
    render_product = rep.create.render_product(camera_path, (width, height))
    rgb_annot = rep.annotators.get("rgb")
    depth_annot = rep.annotators.get("distance_to_image_plane")
    rgb_annot.attach(render_product)
    depth_annot.attach(render_product)

    for frame_idx in range(max(warmup_frames, 1)):
        await rep.orchestrator.step_async()
        if frame_idx % 25 == 0:
            print(f"[export_rgbd] replicator warmup {frame_idx}", flush=True)

    rgb = rgb_annot.get_data()
    depth = depth_annot.get_data()
    print(
        f"[export_rgbd] replicator rgb type={type(rgb)} shape={getattr(rgb, 'shape', None)} dtype={getattr(rgb, 'dtype', None)}",
        flush=True,
    )
    print(
        f"[export_rgbd] replicator depth type={type(depth)} shape={getattr(depth, 'shape', None)} dtype={getattr(depth, 'dtype', None)}",
        flush=True,
    )
    rgb_annot.detach()
    depth_annot.detach()
    render_product.destroy()
    return rgb, depth


def main():
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[export_rgbd] opening stage: {args.usd}", flush=True)
    open_stage(args.usd)
    print("[export_rgbd] stage open requested", flush=True)
    for _ in range(20):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    print(f"[export_rgbd] stage object: {stage}", flush=True)
    if args.robot_base is not None:
        print(f"[export_rgbd] setting robot base: {args.robot_base}", flush=True)
        set_robot_pose(stage, *args.robot_base)
    if args.lift is not None:
        print(f"[export_rgbd] setting lift: {args.lift}", flush=True)
        set_lift(stage, args.lift)
    prim = stage.GetPrimAtPath(args.camera)
    print(f"[export_rgbd] camera prim valid={prim.IsValid()} path={args.camera}", flush=True)
    if not prim.IsValid():
        raise RuntimeError(f"camera prim not found: {args.camera}")

    print("[export_rgbd] creating world/camera wrapper", flush=True)
    world = World(stage_units_in_meters=1.0)
    camera = Camera(
        prim_path=args.camera,
        name="graspnet_export_camera",
        resolution=(args.width, args.height),
        frequency=30,
    )
    print("[export_rgbd] world reset begin", flush=True)
    world.reset()
    print("[export_rgbd] world reset done", flush=True)
    print("[export_rgbd] camera initialize begin", flush=True)
    camera.initialize()
    print("[export_rgbd] camera initialize done", flush=True)
    print("[export_rgbd] attach rgb/depth begin", flush=True)
    camera.add_rgb_to_frame()
    camera.add_distance_to_image_plane_to_frame()
    camera.resume()
    print("[export_rgbd] attach rgb/depth done", flush=True)

    print("[export_rgbd] waiting for camera current_frame rgb/depth", flush=True)
    rgb_value = None
    depth_value = None
    for frame_idx in range(1000):
        world.step(render=True)
        if frame_idx < args.warmup_frames:
            time.sleep(0.02)
            continue
        current_frame = camera.get_current_frame(clone=True)
        rgb_value = current_frame.get("rgb")
        depth_value = current_frame.get("distance_to_image_plane")
        if frame_idx % 50 == 0:
            print(
                f"[export_rgbd] wait {frame_idx}: rgb={getattr(rgb_value, 'shape', None)} depth={getattr(depth_value, 'shape', None)}",
                flush=True,
            )
        if (
            rgb_value is not None
            and depth_value is not None
            and getattr(rgb_value, "size", 0) > 0
            and getattr(depth_value, "size", 0) > 0
        ):
            break
        time.sleep(0.02)
    if rgb_value is None or depth_value is None:
        raise RuntimeError("Camera did not produce rgb/depth data")
    print(
        f"[export_rgbd] rgb raw type={type(rgb_value)} shape={getattr(rgb_value, 'shape', None)} dtype={getattr(rgb_value, 'dtype', None)}",
        flush=True,
    )
    print(
        f"[export_rgbd] depth raw type={type(depth_value)} shape={getattr(depth_value, 'shape', None)} dtype={getattr(depth_value, 'dtype', None)}",
        flush=True,
    )
    rgb = np.array(rgb_value)
    print(f"[export_rgbd] rgb np shape={rgb.shape} dtype={rgb.dtype}", flush=True)
    depth = np.array(depth_value)
    print(f"[export_rgbd] depth np shape={depth.shape} dtype={depth.dtype}", flush=True)

    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    print(f"[export_rgbd] rgb trimmed shape={rgb.shape} dtype={rgb.dtype}", flush=True)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
    print("[export_rgbd] rgb converted", flush=True)

    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0.0] = 0.0
    depth_mm = np.clip(depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    workspace_mask = (depth_mm > 0).astype(np.uint8) * 255
    print("[export_rgbd] depth/mask converted", flush=True)
    intrinsic = camera_intrinsic_matrix(prim, args.width, args.height)
    cv_to_world = camera_cv_to_world_matrix(prim)
    print(f"[export_rgbd] intrinsic={intrinsic}", flush=True)
    print(f"[export_rgbd] camera_cv_to_world={cv_to_world}", flush=True)

    print("[export_rgbd] saving images", flush=True)
    Image.fromarray(rgb).save(os.path.join(args.output_dir, "color.png"))
    Image.fromarray(depth_mm).save(os.path.join(args.output_dir, "depth.png"))
    Image.fromarray(workspace_mask).save(os.path.join(args.output_dir, "workspace_mask.png"))
    print("[export_rgbd] saving meta.mat", flush=True)
    scio.savemat(
        os.path.join(args.output_dir, "meta.mat"),
        {
            "intrinsic_matrix": intrinsic,
            "camera_cv_to_world": cv_to_world,
            "factor_depth": np.array([[1000.0]], dtype=np.float32),
            "camera_prim_path": np.array([args.camera], dtype=object),
            "capture_time": np.array([[time.time()]], dtype=np.float64),
        },
    )
    print(f"[export_rgbd] rgb shape={rgb.shape} depth_m range=({depth[depth > 0].min() if np.any(depth > 0) else 0:.4f}, {depth.max():.4f})")
    print(f"[export_rgbd] saved GraspNet input to {args.output_dir}", flush=True)


try:
    main()
finally:
    simulation_app.close()
