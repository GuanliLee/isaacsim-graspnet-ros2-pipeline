#!/usr/bin/env python3
import json
import math
import os
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


IK_PATHS = (
    Path(os.environ.get("IK_PATH", "/tmp/left_arm_graspnet_ik.json")),
)
MAX_EXECUTION_EE_ANGLE_DEG = float(os.environ.get("MAX_EXECUTION_EE_ANGLE_DEG", "30"))


def load_plan():
    for path in IK_PATHS:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                ik = json.load(f)
            if not ik.get("converged", False):
                raise RuntimeError(
                    f"{path} GraspNet pose IK did not converge; refusing unsafe motion"
                )
            opening_weight = float(ik.get("weights", {}).get("opening_axis", 0.0))
            opening_error = float(ik.get("opening_axis_error", float("inf")))
            if opening_weight <= 0.0 or opening_error >= 0.35:
                raise RuntimeError(
                    f"{path} has unsafe free-wrist orientation "
                    f"(opening weight={opening_weight}, error={opening_error:.3f}); "
                    "re-run search_left_arm_graspnet_pose.py with opening-axis constraint"
                )
            ee_angles = {
                "grasp": ik.get("ee_angle_from_reference_deg"),
                "pregrasp": ik.get("pregrasp_ee_angle_from_reference_deg"),
                "retract": ik.get("retract_ee_angle_from_reference_deg"),
            }
            if any(value is None for value in ee_angles.values()):
                raise RuntimeError(
                    f"{path} has no initial-EE orientation audit; refusing motion"
                )
            over_limit = {
                name: float(value)
                for name, value in ee_angles.items()
                if float(value) > MAX_EXECUTION_EE_ANGLE_DEG
            }
            if over_limit:
                raise RuntimeError(
                    f"{path} exceeds the initial EE +/-{MAX_EXECUTION_EE_ANGLE_DEG:g} "
                    f"deg limit: {over_limit}"
                )
            trajectory_ok = ik.get("trajectory_converged", {})
            if not all(trajectory_ok.get(name, False) for name in ("pregrasp", "grasp", "retract")):
                raise RuntimeError(
                    f"{path} pregrasp/grasp/retract IK is incomplete; refusing unsafe motion"
                )
            q_grasp = [float(v) for v in ik.get("q_grasp", ik["q"])[:6]]
            if "q_pregrasp" not in ik or "q_retract" not in ik:
                raise RuntimeError(
                    f"{path} is an old IK result without q_pregrasp/q_retract; "
                    "run search_left_arm_graspnet_pose.py again"
                )
            q_pre = [float(v) for v in ik["q_pregrasp"][:6]]
            q_retract = [float(v) for v in ik["q_retract"][:6]]
            base = tuple(float(v) for v in ik.get("base", (0.15, -0.27, 0.315, math.pi)))
            lift = float(ik.get("lift", -0.60))
            return base, lift, q_pre, q_grasp, q_retract
    raise FileNotFoundError("no GraspNet IK result found")


BASE, LIFT, Q_PRE, Q_GRASP, Q_RETRACT = load_plan()
PUBLISH_BASE_TARGET = os.environ.get("PUBLISH_BASE_TARGET", "1") == "1"
CURRENT_STATE_TOPIC = "/left_arm/current_joint_states"
CURRENT_STATE_TIMEOUT = float(os.environ.get("CURRENT_STATE_TIMEOUT", "5.0"))
MAX_START_JOINT_DELTA = math.radians(
    float(os.environ.get("MAX_START_JOINT_DELTA_DEG", "120.0"))
)
MAX_COMMAND_STEP = math.radians(float(os.environ.get("MAX_COMMAND_STEP_DEG", "1.0")))


class CurrentJointState:
    def __init__(self):
        self.q = None
        self.received_at = None

    def callback(self, msg):
        by_name = dict(zip(msg.name, msg.position))
        names = [f"left_joint{i}" for i in range(1, 7)]
        if not all(name in by_name for name in names):
            return
        q = [float(by_name[name]) for name in names]
        if all(math.isfinite(value) for value in q):
            self.q = q
            self.received_at = time.monotonic()


class CurrentBaseState:
    def __init__(self):
        self.pose = None

    def callback(self, msg):
        q = msg.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.pose = [
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            float(yaw),
        ]


def wait_for_current_q(node, state):
    deadline = time.monotonic() + CURRENT_STATE_TIMEOUT
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if state.q is not None:
            return list(state.q)
    raise RuntimeError(
        f"no current left-arm state received from {CURRENT_STATE_TOPIC}; "
        "reload ros_topic_robot_control.py before executing"
    )


def wait_for_current_base(node, state):
    deadline = time.monotonic() + CURRENT_STATE_TIMEOUT
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if state.pose is not None:
            return list(state.pose)
    raise RuntimeError("no current robot pose received from /debug/robot_pose")


def validate_start_transition(current_q, target_q):
    deltas = [abs(target - current) for current, target in zip(current_q, target_q)]
    if max(deltas) > MAX_START_JOINT_DELTA:
        raise RuntimeError(
            "refusing unsafe current->pregrasp transition; max joint delta is "
            f"{math.degrees(max(deltas)):.1f} deg"
        )


def wait_until_reached(node, state, target_q, phase):
    tolerance = math.radians(float(os.environ.get("JOINT_REACHED_TOL_DEG", "5.0")))
    timeout = float(os.environ.get("JOINT_REACHED_TIMEOUT", "4.0"))
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if state.q is None:
            continue
        errors = [
            abs(actual - target)
            for actual, target in zip(state.q, target_q)
        ]
        last_error = errors
        if max(errors) <= tolerance:
            print(
                f"{phase}_reached max_error_deg="
                f"{math.degrees(max(errors)):.2f}",
                flush=True,
            )
            return
    if last_error is None:
        raise RuntimeError(f"{phase}: current joint feedback stopped")
    errors_deg = [round(math.degrees(value), 1) for value in last_error]
    raise RuntimeError(
        f"{phase}: commanded pose was not reached; joint errors deg={errors_deg}. "
        "Stopping before the next grasp phase."
    )


def make_base_msg(node, pose=None):
    pose = BASE if pose is None else pose
    msg = PoseStamped()
    msg.header.frame_id = "world"
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.position.x = pose[0]
    msg.pose.position.y = pose[1]
    msg.pose.position.z = pose[2]
    msg.pose.orientation.z = math.sin(pose[3] * 0.5)
    msg.pose.orientation.w = math.cos(pose[3] * 0.5)
    return msg


def publish_base(node, pub, seconds, hz=20, pose=None):
    if not PUBLISH_BASE_TARGET:
        time.sleep(max(0.0, seconds))
        return
    for _ in range(int(seconds * hz)):
        pub.publish(make_base_msg(node, pose))
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(1.0 / hz)


def publish_enable(pub):
    msg = Bool()
    msg.data = True
    pub.publish(msg)


def publish_arm(
    node, pub, q, gripper, seconds, hz=20, enable_pub=None, base_pub=None,
    base_pose=None,
):
    msg = JointState()
    msg.name = [f"left_joint{i}" for i in range(1, 8)] + ["lifting_joint"]
    msg.position = list(q) + [float(gripper), LIFT]
    for _ in range(int(seconds * hz)):
        msg.header.stamp = node.get_clock().now().to_msg()
        if enable_pub is not None:
            publish_enable(enable_pub)
        if base_pub is not None:
            if PUBLISH_BASE_TARGET:
                base_pub.publish(make_base_msg(node, base_pose))
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(1.0 / hz)


def publish_arm_trajectory(
    node,
    pub,
    start_q,
    end_q,
    start_gripper,
    end_gripper,
    seconds,
    hz=40,
    enable_pub=None,
    base_pub=None,
    base_pose=None,
):
    msg = JointState()
    msg.name = [f"left_joint{i}" for i in range(1, 8)] + ["lifting_joint"]
    largest_delta = max(
        [abs(b - a) for a, b in zip(start_q, end_q)]
        + [abs(end_gripper - start_gripper)]
    )
    steps = max(1, int(seconds * hz), int(math.ceil(largest_delta / MAX_COMMAND_STEP)))
    for index in range(steps + 1):
        t = index / steps
        smooth = t * t * (3.0 - 2.0 * t)
        q = [a + (b - a) * smooth for a, b in zip(start_q, end_q)]
        gripper = start_gripper + (end_gripper - start_gripper) * smooth
        msg.position = list(q) + [float(gripper), LIFT]
        msg.header.stamp = node.get_clock().now().to_msg()
        if enable_pub is not None:
            publish_enable(enable_pub)
        if base_pub is not None and PUBLISH_BASE_TARGET:
            base_pub.publish(make_base_msg(node, base_pose))
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(1.0 / hz)


def interpolate_angle(start, end, t):
    delta = math.atan2(math.sin(end - start), math.cos(end - start))
    return start + delta * t


def publish_base_trajectory(
    node,
    base_pub,
    start_pose,
    end_pose,
    seconds,
    enable_pub,
    arm_pub=None,
    hold_q=None,
    gripper=1.0,
    hz=50,
):
    steps = max(1, int(seconds * hz))
    arm_msg = JointState()
    arm_msg.name = [f"left_joint{i}" for i in range(1, 8)] + ["lifting_joint"]
    if hold_q is not None:
        arm_msg.position = list(hold_q) + [float(gripper), LIFT]
    for index in range(steps + 1):
        t = index / steps
        smooth = t * t * (3.0 - 2.0 * t)
        pose = [
            start_pose[i] + (end_pose[i] - start_pose[i]) * smooth
            for i in range(3)
        ]
        pose.append(interpolate_angle(start_pose[3], end_pose[3], smooth))
        publish_enable(enable_pub)
        base_pub.publish(make_base_msg(node, pose))
        if arm_pub is not None and hold_q is not None:
            arm_msg.header.stamp = node.get_clock().now().to_msg()
            arm_pub.publish(arm_msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(1.0 / hz)


def main():
    rclpy.init()
    node = rclpy.create_node("left_graspnet_executor")
    enable_pub = node.create_publisher(Bool, "/enable_flag", 10)
    base_pub = node.create_publisher(PoseStamped, "/base_target_pose", 10)
    arm_pub = node.create_publisher(JointState, "/joint_states_gripper_l", 10)
    current_state = CurrentJointState()
    current_base_state = CurrentBaseState()
    node.create_subscription(
        JointState, CURRENT_STATE_TOPIC, current_state.callback, 10
    )
    node.create_subscription(
        PoseStamped, "/debug/robot_pose", current_base_state.callback, 10
    )

    print("step0: wait for actual left-arm joint state", flush=True)
    current_q = wait_for_current_q(node, current_state)
    current_base = wait_for_current_base(node, current_base_state)
    validate_start_transition(current_q, Q_PRE)
    print(
        "current_q_deg:",
        [round(math.degrees(value), 1) for value in current_q],
        flush=True,
    )

    dx = current_base[0] - BASE[0]
    dy = current_base[1] - BASE[1]
    distance = math.hypot(dx, dy)
    staging_distance = float(os.environ.get("STAGING_DISTANCE_M", "0.45"))
    if distance < 0.10:
        direction_x, direction_y = 1.0, 0.0
    else:
        direction_x, direction_y = dx / distance, dy / distance
    staging_base = [
        BASE[0] + staging_distance * direction_x,
        BASE[1] + staging_distance * direction_y,
        BASE[2],
        BASE[3],
    ]

    print("step1: move base to shelf staging pose with arm unchanged", flush=True)
    publish_base_trajectory(
        node,
        base_pub,
        current_base,
        staging_base,
        float(os.environ.get("BASE_TO_STAGING_SECONDS", "6.0")),
        enable_pub,
        arm_pub=arm_pub,
        hold_q=current_q,
        gripper=1.0,
    )

    print("step2: open gripper and form pregrasp outside the shelf", flush=True)
    publish_arm_trajectory(
        node,
        arm_pub,
        current_q,
        Q_PRE,
        1.0,
        1.0,
        float(os.environ.get("START_TO_PREGRASP_SECONDS", "8.0")),
        hz=50,
        enable_pub=enable_pub,
        base_pub=base_pub,
        base_pose=staging_base,
    )
    wait_until_reached(node, current_state, Q_PRE, "pregrasp")

    print("step3: move base from staging pose to final pregrasp pose", flush=True)
    publish_base_trajectory(
        node,
        base_pub,
        staging_base,
        BASE,
        float(os.environ.get("STAGING_TO_GRASP_SECONDS", "6.0")),
        enable_pub,
        arm_pub=arm_pub,
        hold_q=Q_PRE,
        gripper=1.0,
    )
    wait_until_reached(node, current_state, Q_PRE, "pregrasp_at_shelf")
    publish_base(node, base_pub, 0.5)

    print("step4: smoothly approach GraspNet IK target with left arm", flush=True)
    publish_arm_trajectory(
        node,
        arm_pub,
        Q_PRE,
        Q_GRASP,
        1.0,
        1.0,
        6.0,
        hz=50,
        enable_pub=enable_pub,
        base_pub=base_pub,
    )
    wait_until_reached(node, current_state, Q_GRASP, "grasp")
    publish_base(node, base_pub, 0.5)

    print("step5: close gripper", flush=True)
    publish_arm(node, arm_pub, Q_GRASP, -1.0, 5.0, hz=40, enable_pub=enable_pub, base_pub=base_pub)
    publish_base(node, base_pub, 0.5)

    print("step6: smoothly retract while closed", flush=True)
    publish_arm_trajectory(
        node,
        arm_pub,
        Q_GRASP,
        Q_RETRACT,
        -1.0,
        -1.0,
        5.0,
        hz=50,
        enable_pub=enable_pub,
        base_pub=base_pub,
    )
    wait_until_reached(node, current_state, Q_RETRACT, "retract")
    publish_base(node, base_pub, 0.5)

    print("done", flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
