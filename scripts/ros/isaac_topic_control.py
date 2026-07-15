import json
import math
import os
import sys
import time

import omni.graph.core as og
import omni.usd
from pxr import Gf, UsdGeom

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ["ROS_DOMAIN_ID"] = os.environ.get("ISAAC_ROS_DOMAIN_ID", "44")
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "SUBNET")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")

STATUS_PATH = "/tmp/isaac_ros_topic_robot_control_status.txt"


def _write_status(message):
    try:
        with open(STATUS_PATH, "a", encoding="utf-8") as f:
            f.write(f"{time.time():.3f} {message}\n")
    except Exception:
        pass


_write_status(
    "module_import "
    f"domain={os.environ.get('ROS_DOMAIN_ID')} "
    f"range={os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE')}"
)

_ISAAC_PIP_PREBUNDLES = (
    (
        "/isaac-sim/extscache/"
        "omni.services.pip_archive-0.16.0+107.0.3.lx64.cp311/"
        "pip_prebundle"
    ),
    (
        "/isaac-sim/extscache/"
        "omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/"
        "pip_prebundle"
    ),
)
for _prebundle in _ISAAC_PIP_PREBUNDLES:
    if os.path.isdir(_prebundle) and _prebundle not in sys.path:
        sys.path.append(_prebundle)

ROBOT_PATH = "/World/Robot"
JOINTS_PATH = f"{ROBOT_PATH}/joints"
TARGET_PATHS = tuple(
    f"/World/Objects/Shelf_Beverages/AD_Calcium_Milk/AD_Calcium_Milk_{idx:02d}"
    for idx in range(1, 10)
)
DEBUG_TARGET_PATH = "/World/Objects/Shelf_Beverages/AD_Calcium_Milk/AD_Calcium_Milk_03"

LEFT_ARM_JOINTS = [f"left_joint{i}" for i in range(1, 7)]
RIGHT_ARM_JOINTS = [f"right_joint{i}" for i in range(1, 7)]
LEFT_GRIPPER_JOINTS = ("left_joint7", "left_joint8")
RIGHT_GRIPPER_JOINTS = ("right_joint7", "right_joint8")
LEFT_WRIST_PATH = f"{ROBOT_PATH}/left_link6"
LEFT_FINGER_PATHS = (f"{ROBOT_PATH}/left_link7", f"{ROBOT_PATH}/left_link8")
RIGHT_WRIST_PATH = f"{ROBOT_PATH}/right_link6"
RIGHT_FINGER_PATHS = (f"{ROBOT_PATH}/right_link7", f"{ROBOT_PATH}/right_link8")
LEFT_TCP_EXTRINSIC_PATH = os.environ.get(
    "LEFT_TCP_EXTRINSIC_PATH",
    "/workspace/grasp-pipeline/config/tcp_extrinsic.json",
)

MAX_GRIPPER_OPEN_M = 0.085
GRIPPER_CLOSE_M = 0.0
GRIPPER_DRIVE_STIFFNESS = 1000.0
GRIPPER_DRIVE_DAMPING = 50.0
GRIPPER_DRIVE_MAX_FORCE = 1000.0
GRASP_ATTACH_DISTANCE_M = 0.055
ARM_TARGET_SMOOTH_ALPHA = 0.25
CMD_TIMEOUT_SEC = 0.5

_node = None
_robot_pose_pub = None
_target_pose_pub = None
_left_state_pub = None
_enabled = False
_left_cmd = None
_right_cmd = None
_lift_cmd = None
_left_applied_cmd = None
_right_applied_cmd = None
_cmd_vel = None
_base_target_pose = None
_last_left_time = 0.0
_last_right_time = 0.0
_last_cmd_vel_time = 0.0
_last_base_target_time = 0.0
_last_update_time = None
_last_debug_pose_time = 0.0
_grasp_attached_side = None
_grasp_gripper_to_target = None
_grasp_target_path = None
_dynamic_control = None
_robot_articulation_handle = None
_dynamic_control_disabled = False


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _set_attr(stage, prim_path, attr_name, value):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    attr = prim.GetAttribute(attr_name)
    if not attr.IsValid():
        return False
    attr.Set(value)
    return True


def _get_world_position(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    return matrix.ExtractTranslation()


def _set_world_position_by_delta(stage, prim_path, desired_world):
    prim = stage.GetPrimAtPath(prim_path)
    current_world = _get_world_position(stage, prim_path)
    if not prim.IsValid() or current_world is None:
        return False
    delta = Gf.Vec3d(
        float(desired_world[0] - current_world[0]),
        float(desired_world[1] - current_world[1]),
        float(desired_world[2] - current_world[2]),
    )
    xform = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            translate_op = op
            break
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    local = translate_op.Get() or Gf.Vec3d(0.0, 0.0, 0.0)
    translate_op.Set(
        Gf.Vec3d(
            float(local[0] + delta[0]),
            float(local[1] + delta[1]),
            float(local[2] + delta[2]),
        )
    )
    return True


def _finger_center(stage, finger_paths):
    first = _get_world_position(stage, finger_paths[0])
    second = _get_world_position(stage, finger_paths[1])
    if first is None or second is None:
        return None
    return Gf.Vec3d(
        float((first[0] + second[0]) * 0.5),
        float((first[1] + second[1]) * 0.5),
        float((first[2] + second[2]) * 0.5),
    )


def _load_left_tcp_translation():
    try:
        with open(LEFT_TCP_EXTRINSIC_PATH, "r", encoding="utf-8") as handle:
            values = json.load(handle)["link6_tcp"]["translation_m"]
        if len(values) == 3 and all(math.isfinite(float(value)) for value in values):
            return Gf.Vec3d(*(float(value) for value in values))
    except Exception as exc:
        _write_status(f"tcp_extrinsic_load_error {exc!r}")
    return None


LEFT_TCP_TRANSLATION = None


def _left_tcp_center(stage):
    global LEFT_TCP_TRANSLATION
    if LEFT_TCP_TRANSLATION is None:
        LEFT_TCP_TRANSLATION = _load_left_tcp_translation()
    if LEFT_TCP_TRANSLATION is None:
        return None
    prim = stage.GetPrimAtPath(LEFT_WRIST_PATH)
    if not prim.IsValid():
        return None
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    return matrix.Transform(LEFT_TCP_TRANSLATION)


def _distance(a, b):
    if a is None or b is None:
        return float("inf")
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def _closest_target_path(stage, reference_pos):
    best_path = None
    best_pos = None
    best_distance = float("inf")
    for path in TARGET_PATHS:
        target_pos = _get_world_position(stage, path)
        distance = _distance(reference_pos, target_pos)
        if distance < best_distance:
            best_path = path
            best_pos = target_pos
            best_distance = distance
    return best_path, best_pos, best_distance


def _publish_pose(pub, frame_id, position, yaw=0.0):
    if pub is None or position is None:
        return

    from geometry_msgs.msg import PoseStamped

    msg = PoseStamped()
    msg.header.frame_id = "world"
    if _node is not None:
        msg.header.stamp = _node.get_clock().now().to_msg()
    msg.pose.position.x = float(position[0])
    msg.pose.position.y = float(position[1])
    msg.pose.position.z = float(position[2])
    msg.pose.orientation.z = math.sin(yaw * 0.5)
    msg.pose.orientation.w = math.cos(yaw * 0.5)
    pub.publish(msg)


def _get_robot_pose(stage):
    translate_op, rotate_z_op = _get_xform_ops(stage, ROBOT_PATH)
    if translate_op is None or rotate_z_op is None:
        return None, 0.0

    position = translate_op.Get() or Gf.Vec3d(2.525, 0.225, 0.315)
    yaw_deg = rotate_z_op.Get()
    if yaw_deg is None:
        yaw_deg = 180.0
    return position, math.radians(yaw_deg)


def _yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _get_xform_ops(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None, None

    xform = UsdGeom.Xformable(prim)
    translate_op = None
    rotate_z_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            translate_op = op
        elif op.GetOpName() == "xformOp:rotateZ":
            rotate_z_op = op

    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    if rotate_z_op is None:
        rotate_z_op = xform.AddRotateZOp()

    return translate_op, rotate_z_op


def _set_robot_pose(stage, pose):
    if pose is None:
        return

    translate_op, rotate_z_op = _get_xform_ops(stage, ROBOT_PATH)
    if translate_op is None or rotate_z_op is None:
        return

    x, y, z, yaw = pose
    translate_op.Set(Gf.Vec3d(x, y, z))
    rotate_z_op.Set(math.degrees(yaw))


def _set_revolute_target(stage, joint_name, radians):
    path = f"{JOINTS_PATH}/{joint_name}"
    return _set_attr(stage, path, "drive:angular:physics:targetPosition", math.degrees(radians))


def _get_revolute_position(stage, joint_name):
    global _dynamic_control, _robot_articulation_handle, _dynamic_control_disabled

    try:
        if _dynamic_control_disabled:
            raise RuntimeError("dynamic control lookup disabled")
        if _dynamic_control is None:
            from omni.isaac.dynamic_control import _dynamic_control as dc_module

            _dynamic_control = dc_module.acquire_dynamic_control_interface()
        if not _robot_articulation_handle:
            _robot_articulation_handle = _dynamic_control.get_articulation(ROBOT_PATH)
            if not _robot_articulation_handle:
                _dynamic_control_disabled = True
                _write_status(
                    "dynamic_control_articulation_missing; using USD joint state fallback"
                )
        if _robot_articulation_handle:
            dof = _dynamic_control.find_articulation_dof(
                _robot_articulation_handle, joint_name
            )
            if dof:
                state = _dynamic_control.get_dof_state(
                    dof, _dynamic_control.STATE_POS
                )
                if state is not None and math.isfinite(float(state.pos)):
                    return float(state.pos)
    except Exception:
        pass

    prim = stage.GetPrimAtPath(f"{JOINTS_PATH}/{joint_name}")
    if not prim.IsValid():
        return None
    for attr_name in (
        "state:angular:physics:position",
        "drive:angular:physics:targetPosition",
    ):
        attr = prim.GetAttribute(attr_name)
        value = attr.Get() if attr.IsValid() else None
        if value is not None:
            return math.radians(float(value))
    return None


def _publish_left_joint_state(stage):
    if _left_state_pub is None:
        return

    positions = [_get_revolute_position(stage, name) for name in LEFT_ARM_JOINTS]
    if any(value is None or not math.isfinite(value) for value in positions):
        return

    from sensor_msgs.msg import JointState

    msg = JointState()
    msg.header.stamp = _node.get_clock().now().to_msg()
    msg.name = list(LEFT_ARM_JOINTS)
    msg.position = [float(value) for value in positions]
    _left_state_pub.publish(msg)


def _set_prismatic_target(stage, joint_name, meters):
    path = f"{JOINTS_PATH}/{joint_name}"
    return _set_attr(stage, path, "drive:linear:physics:targetPosition", meters)


def _configure_gripper_joint(stage, joint_name, lower_limit, upper_limit):
    path = f"{JOINTS_PATH}/{joint_name}"
    _set_attr(stage, path, "physics:lowerLimit", lower_limit)
    _set_attr(stage, path, "physics:upperLimit", upper_limit)
    _set_attr(stage, path, "drive:linear:physics:stiffness", GRIPPER_DRIVE_STIFFNESS)
    _set_attr(stage, path, "drive:linear:physics:damping", GRIPPER_DRIVE_DAMPING)
    _set_attr(stage, path, "drive:linear:physics:maxForce", GRIPPER_DRIVE_MAX_FORCE)


def _set_gripper_target(stage, gripper_joints, command_value):
    _configure_gripper_joint(stage, gripper_joints[0], 0.0, MAX_GRIPPER_OPEN_M)
    _configure_gripper_joint(stage, gripper_joints[1], -MAX_GRIPPER_OPEN_M, 0.0)

    gripper = _gripper_command_to_width(command_value)
    _set_prismatic_target(stage, gripper_joints[0], gripper)
    _set_prismatic_target(stage, gripper_joints[1], -gripper)


def _gripper_command_to_width(command_value):
    return MAX_GRIPPER_OPEN_M if command_value > 0.0 else GRIPPER_CLOSE_M


def _smooth_arm_command(previous, command):
    if command is None or len(command) < 7:
        return previous

    command = [float(value) for value in command[:7]]
    if previous is None or len(previous) < 7:
        return command

    smoothed = []
    for prev_value, target_value in zip(previous[:6], command[:6]):
        smoothed.append(
            prev_value + ARM_TARGET_SMOOTH_ALPHA * (target_value - prev_value)
        )
    smoothed.append(command[6])
    return smoothed


def _apply_arm(stage, prefix_joints, gripper_joints, command, previous_command):
    command = _smooth_arm_command(previous_command, command)
    if command is None or len(command) < 7:
        return previous_command

    for joint_name, joint_position in zip(prefix_joints, command[:6]):
        _set_revolute_target(stage, joint_name, joint_position)

    _set_gripper_target(stage, gripper_joints, command[6])
    return command


def _update_grasp_attachment(stage):
    global _grasp_attached_side, _grasp_gripper_to_target, _grasp_target_path

    candidates = (
        ("left", _left_applied_cmd, LEFT_WRIST_PATH, LEFT_FINGER_PATHS),
        ("right", _right_applied_cmd, RIGHT_WRIST_PATH, RIGHT_FINGER_PATHS),
    )

    if _grasp_attached_side is None:
        for side, command, wrist_path, finger_paths in candidates:
            if command is None or len(command) < 7 or command[6] > 0.0:
                continue
            finger_pos = (
                _left_tcp_center(stage)
                if side == "left"
                else _finger_center(stage, finger_paths)
            )
            wrist_pos = _get_world_position(stage, wrist_path)
            if wrist_pos is None or finger_pos is None:
                continue
            target_path, target_pos, distance = _closest_target_path(stage, finger_pos)
            if target_path is None or distance > GRASP_ATTACH_DISTANCE_M:
                continue
            _grasp_attached_side = side
            _grasp_target_path = target_path
            _grasp_gripper_to_target = Gf.Vec3d(
                float(target_pos[0] - finger_pos[0]),
                float(target_pos[1] - finger_pos[1]),
                float(target_pos[2] - finger_pos[2]),
            )
            break

    if _grasp_attached_side is None or _grasp_gripper_to_target is None:
        return

    for side, command, wrist_path, _finger_paths in candidates:
        if side != _grasp_attached_side:
            continue
        if command is not None and len(command) >= 7 and command[6] > 0.0:
            _grasp_attached_side = None
            _grasp_gripper_to_target = None
            _grasp_target_path = None
            return
        finger_pos = (
            _left_tcp_center(stage)
            if side == "left"
            else _finger_center(stage, _finger_paths)
        )
        if finger_pos is not None and _grasp_target_path is not None:
            _set_world_position_by_delta(
                stage, _grasp_target_path, finger_pos + _grasp_gripper_to_target
            )
        return


def _integrate_cmd_vel(stage, dt):
    global _cmd_vel

    if _cmd_vel is None:
        return

    vx, vy, wz = _cmd_vel
    translate_op, rotate_z_op = _get_xform_ops(stage, ROBOT_PATH)
    if translate_op is None or rotate_z_op is None:
        return

    pos = translate_op.Get() or Gf.Vec3d(2.525, 0.225, 0.315)
    yaw_deg = rotate_z_op.Get()
    if yaw_deg is None:
        yaw_deg = 180.0

    yaw = math.radians(yaw_deg)
    world_vx = math.cos(yaw) * vx - math.sin(yaw) * vy
    world_vy = math.sin(yaw) * vx + math.cos(yaw) * vy

    translate_op.Set(Gf.Vec3d(pos[0] + world_vx * dt, pos[1] + world_vy * dt, pos[2]))
    rotate_z_op.Set(yaw_deg + math.degrees(wz * dt))


def _init_ros(db):
    global _node, _robot_pose_pub, _target_pose_pub, _left_state_pub

    if _node is not None:
        return True

    try:
        _write_status("init_ros_import_start")
        from geometry_msgs.msg import PoseStamped
        import rclpy
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool
    except Exception as exc:
        _write_status(f"init_ros_import_error {exc!r}")
        db.log_error(f"ROS2 Python modules are not available in Isaac Sim: {exc}")
        return False

    if rclpy.ok():
        try:
            _write_status("rclpy_shutdown_for_domain_reset")
            rclpy.shutdown()
        except Exception as exc:
            _write_status(f"rclpy_shutdown_error {exc!r}")

    if not rclpy.ok():
        _write_status("rclpy_init_start")
        rclpy.init(args=None)

    _node = rclpy.create_node("isaac_ros_topic_robot_control")
    _write_status("node_created isaac_ros_topic_robot_control")

    def on_enable(msg):
        global _enabled
        _enabled = bool(msg.data)

    def on_left(msg):
        global _left_cmd, _lift_cmd, _last_left_time
        _left_cmd = list(msg.position[:7])
        if msg.name and "lifting_joint" in msg.name:
            lift_index = list(msg.name).index("lifting_joint")
            if lift_index < len(msg.position):
                _lift_cmd = float(msg.position[lift_index])
        _last_left_time = time.monotonic()

    def on_right(msg):
        global _right_cmd, _last_right_time
        _right_cmd = list(msg.position[:7])
        _last_right_time = time.monotonic()

    def on_cmd_vel(msg):
        global _cmd_vel, _last_cmd_vel_time
        _cmd_vel = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
        _last_cmd_vel_time = time.monotonic()

    def on_base_target_pose(msg):
        global _base_target_pose, _last_base_target_time
        position = msg.pose.position
        yaw = _yaw_from_quaternion(msg.pose.orientation)
        _base_target_pose = (
            float(position.x),
            float(position.y),
            float(position.z),
            yaw,
        )
        _last_base_target_time = time.monotonic()

    _node.create_subscription(Bool, "/enable_flag", on_enable, 10)
    _node.create_subscription(JointState, "/joint_states_gripper_l", on_left, 10)
    _node.create_subscription(JointState, "/joint_states_gripper_r", on_right, 10)
    _node.create_subscription(Twist, "/cmd_vel", on_cmd_vel, 10)
    _node.create_subscription(PoseStamped, "/base_target_pose", on_base_target_pose, 10)
    _robot_pose_pub = _node.create_publisher(PoseStamped, "/debug/robot_pose", 10)
    _target_pose_pub = _node.create_publisher(PoseStamped, "/debug/target_pose", 10)
    _left_state_pub = _node.create_publisher(
        JointState, "/left_arm/current_joint_states", 10
    )
    return True


def setup(db: og.Database):
    _write_status("setup_called")
    _init_ros(db)


def compute(db: og.Database):
    global _left_cmd, _right_cmd, _left_applied_cmd, _right_applied_cmd
    global _cmd_vel, _base_target_pose, _lift_cmd
    global _last_update_time, _last_debug_pose_time

    if not _init_ros(db):
        _write_status("compute_init_failed")
        return False

    import rclpy

    rclpy.spin_once(_node, timeout_sec=0.0)

    now = time.monotonic()
    if now - _last_left_time > CMD_TIMEOUT_SEC:
        _left_cmd = None
    if now - _last_right_time > CMD_TIMEOUT_SEC:
        _right_cmd = None
    if now - _last_cmd_vel_time > CMD_TIMEOUT_SEC:
        _cmd_vel = None
    if now - _last_base_target_time > CMD_TIMEOUT_SEC:
        _base_target_pose = None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return False

    if _enabled:
        if _lift_cmd is not None:
            _set_prismatic_target(stage, "lifting_joint", _lift_cmd)
        _left_applied_cmd = _apply_arm(
            stage,
            LEFT_ARM_JOINTS,
            LEFT_GRIPPER_JOINTS,
            _left_cmd,
            _left_applied_cmd,
        )
        _right_applied_cmd = _apply_arm(
            stage,
            RIGHT_ARM_JOINTS,
            RIGHT_GRIPPER_JOINTS,
            _right_cmd,
            _right_applied_cmd,
        )
        _update_grasp_attachment(stage)

    if now - _last_debug_pose_time > 0.2:
        _publish_left_joint_state(stage)
        robot_position, robot_yaw = _get_robot_pose(stage)
        _publish_pose(_robot_pose_pub, "robot", robot_position, robot_yaw)
        debug_target = _grasp_target_path or DEBUG_TARGET_PATH
        _publish_pose(_target_pose_pub, "target", _get_world_position(stage, debug_target))
        _last_debug_pose_time = now

    if _base_target_pose is not None:
        _set_robot_pose(stage, _base_target_pose)
    else:
        if _last_update_time is None:
            dt = 1.0 / 60.0
        else:
            dt = _clamp(now - _last_update_time, 0.0, 1.0 / 15.0)
        _integrate_cmd_vel(stage, dt)

    _last_update_time = now

    db.outputs.execOut = og.ExecutionAttributeState.ENABLED
    return True


def cleanup(db: og.Database):
    global _node, _robot_pose_pub, _target_pose_pub, _left_state_pub

    _write_status("cleanup_called")
    if _node is not None:
        _node.destroy_node()
        _node = None
        _robot_pose_pub = None
        _target_pose_pub = None
        _left_state_pub = None
