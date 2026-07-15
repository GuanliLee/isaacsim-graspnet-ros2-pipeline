# Architecture

## 1. Scene and ROS Bridge

The Isaac Sim OmniGraph Script Node loads `scripts/ros/isaac_topic_control.py`.
It receives base and arm commands, publishes current state feedback, and keeps
the configured ROS domain consistent with the ROS 2 container.

The reference implementation expects these prims:

```text
/World/Robot
/World/Robot/joints
/World/Robot/left_link6
/World/Robot/left_link7
/World/Robot/left_link8
/World/Robot/left_link6/left_camera
```

Change the constants in the ROS bridge and the corresponding configuration if
your USD uses different prim paths.

## 2. Perception

`export_rgbd.py` exports RGB, depth, camera intrinsics, and the CV-camera to
world transform. `detect_target.py` detects white AD Calcium Milk candidates
and ranks complete first-row detections near the left gripper image axis.
`segment_target.py` uses the selected bounding box as a SAM prompt.

The raw SAM mask becomes GraspNet's `workspace_mask.png`, so the grasp network
receives only the selected object cloud rather than the full shelf scene.

## 3. Grasp Filtering

`select_grasp.py` rejects candidates that:

- Project outside the selected mask.
- Fall outside the configured vertical part of the detection box.
- Lie near the object bottom or shelf surface.
- Approach from behind the object.
- Put the pregrasp point inside the shelf instead of on the camera side.

The GraspNet raw rotation uses column 0 as approach and column 1 as jaw opening.
The PiPER TCP mapping is applied before IK.

## 4. IK and TCP Alignment

`solve_ik.py` minimizes TCP position, tool approach-axis, and opening-axis
errors with a numerical Jacobian. The TCP position is computed from the link6
pose and `config/tcp_extrinsic.json`.

Two parallel-gripper orientations separated by a 180-degree rotation around
the approach axis are equivalent geometrically. The solver selects the branch
nearest the initial end-effector orientation to avoid unnecessary wrist flips.

The solver produces three validated joint configurations:

```text
pregrasp = grasp - approach * pregrasp_distance
grasp    = selected GraspNet position
retract  = grasp - approach * retract_distance + world_z * retract_lift
```

## 5. Execution

`execute_grasp.py` reads the validated JSON and refuses motion when trajectory
flags, opening-axis constraints, initial-EE orientation checks, or current
joint feedback are missing. It then executes staging, pregrasp, approach,
close, and retract phases with smooth interpolation.

The staging movement is a safety workaround for the absence of a full
collision-aware planner. Set `STAGING_DISTANCE_M=0` only after validating that
the current-to-pregrasp joint-space trajectory is collision-free.
