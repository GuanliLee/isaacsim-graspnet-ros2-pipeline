#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preview}"
if [[ "${MODE}" != "preview" && "${MODE}" != "execute" ]]; then
  echo "usage: $0 [preview|execute]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${PIPELINE_CONFIG:-${REPO_ROOT}/config/pipeline.env}"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

DOMAIN_ID="${ROS_DOMAIN_ID:-44}"
ISAAC_CONTAINER="${ISAAC_CONTAINER:-isaac-sim-5.1.0}"
ROS_CONTAINER="${ROS_CONTAINER:-ros-jazzy-rviz}"
HOST_PYTHON="${HOST_PYTHON:-python3}"
GRASPNET_DIR="${GRASPNET_DIR:-${REPO_ROOT}/third_party/graspnet-baseline}"
GRASPNET_RUNNER="${GRASPNET_RUNNER:-${GRASPNET_DIR}/demo.py}"
GRASPNET_CHECKPOINT="${GRASPNET_CHECKPOINT:-${REPO_ROOT}/model/graspnet/checkpoint-rs.tar}"
YOLOE_WEIGHTS="${YOLOE_WEIGHTS:-${REPO_ROOT}/model/yoloe/yoloe-26l-seg.pt}"
SAM_WEIGHTS="${SAM_WEIGHTS:-${REPO_ROOT}/model/sam/sam_b.pt}"
export PYTHONPATH="${REPO_ROOT}/third_party/ultralytics:${REPO_ROOT}/third_party/graspnetAPI:${PYTHONPATH:-}"

REPO_ISAAC="${REPO_ISAAC:-/workspace/grasp-pipeline}"
USD_ISAAC="${USD_ISAAC:-${REPO_ISAAC}/assets/usd/market2_shelf_background_bundle_portable/new_0713_3camera.usd}"
CAMERA_PRIM="${CAMERA_PRIM:-/World/Robot/left_link6/left_camera}"
TCP_EXTRINSIC_ISAAC="${TCP_EXTRINSIC_ISAAC:-${REPO_ISAAC}/config/tcp_extrinsic.json}"
ROS_EXECUTOR="${ROS_EXECUTOR:-/tmp/execute_grasp.py}"

BASE_X="${BASE_X:-0.50}"
BASE_Y="${BASE_Y:--0.20}"
BASE_Z="${BASE_Z:-0.315}"
BASE_YAW="${BASE_YAW:-3.141592653589793}"
LIFT_POSITION="${LIFT_POSITION:--0.4}"

WORK_HOST="${REPO_ROOT}/work"
CAPTURE_HOST="${WORK_HOST}/capture"
INPUT_HOST="${WORK_HOST}/graspnet_input"
RESULT_HOST="${REPO_ROOT}/outputs/latest"
WORK_ISAAC="${REPO_ISAAC}/work"
CAPTURE_ISAAC="${WORK_ISAAC}/capture"
GRASP_JSON_ISAAC="${WORK_ISAAC}/selected_grasp.json"
IK_ISAAC="/tmp/left_arm_front_ik.json"

mkdir -p "${CAPTURE_HOST}" "${INPUT_HOST}" "${RESULT_HOST}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 1
  fi
}

require_file "${GRASPNET_RUNNER}"
require_file "${GRASPNET_CHECKPOINT}"
require_file "${YOLOE_WEIGHTS}"
require_file "${SAM_WEIGHTS}"

read -r BASE_QZ BASE_QW < <(
  "${HOST_PYTHON}" -c \
    'import math,sys; yaw=float(sys.argv[1]); print(math.sin(yaw/2), math.cos(yaw/2))' \
    "${BASE_YAW}"
)

ros_exec() {
  docker exec "${ROS_CONTAINER}" bash -lc \
    "export ROS_DOMAIN_ID=${DOMAIN_ID}; export ROS_LOCALHOST_ONLY=0; source /opt/ros/jazzy/setup.bash; $*"
}

echo "[1/9] ROS Domain ${DOMAIN_ID} preflight"
ros_exec "ros2 topic info /joint_states_gripper_l -v" \
  | grep -q "isaac_ros_topic_robot_control"

echo "[2/9] Move the live robot to the recognition pose"
ros_exec "ros2 topic pub --once /enable_flag std_msgs/msg/Bool '{data: true}' >/dev/null"
ros_exec "timeout 12 ros2 topic pub -r 20 /base_target_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: world}, pose: {position: {x: ${BASE_X}, y: ${BASE_Y}, z: ${BASE_Z}}, orientation: {z: ${BASE_QZ}, w: ${BASE_QW}}}}' >/dev/null || true"
ros_exec "timeout 4 ros2 topic echo /debug/robot_pose --once" || true

echo "[3/9] Capture left-wrist RGB-D"
docker exec "${ISAAC_CONTAINER}" bash -lc \
  "/isaac-sim/python.sh '${REPO_ISAAC}/scripts/perception/export_rgbd.py' \
    --headless --usd '${USD_ISAAC}' --camera '${CAMERA_PRIM}' \
    --width 640 --height 360 --warmup-frames 80 \
    --robot-base '${BASE_X}' '${BASE_Y}' '${BASE_Z}' '${BASE_YAW}' \
    --lift '${LIFT_POSITION}' --output-dir '${CAPTURE_ISAAC}'"

echo "[4/9] Detect the first-row white AD Calcium Milk bottle"
"${HOST_PYTHON}" "${REPO_ROOT}/scripts/perception/detect_target.py" \
  --image "${CAPTURE_HOST}/color.png" \
  --depth "${CAPTURE_HOST}/depth.png" \
  --weights "${YOLOE_WEIGHTS}" \
  --out-dir "${RESULT_HOST}" \
  --target-mode white_ad_milk --select-strategy first_row_arm --conf 0.01

read -r X1 Y1 X2 Y2 < <(
  "${HOST_PYTHON}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(*d["selected"]["bbox_xyxy"])' \
    "${RESULT_HOST}/yoloe_sam_selected_summary.json"
)

echo "[5/9] Segment the selected detection with SAM"
"${HOST_PYTHON}" "${REPO_ROOT}/scripts/perception/segment_target.py" \
  --image "${CAPTURE_HOST}/color.png" \
  --bbox "${X1}" "${Y1}" "${X2}" "${Y2}" \
  --weights "${SAM_WEIGHTS}" --out-dir "${RESULT_HOST}"

cp "${CAPTURE_HOST}/color.png" "${INPUT_HOST}/color.png"
cp "${CAPTURE_HOST}/depth.png" "${INPUT_HOST}/depth.png"
cp "${CAPTURE_HOST}/meta.mat" "${INPUT_HOST}/meta.mat"
cp "${RESULT_HOST}/sam_mask_selected_raw.png" "${INPUT_HOST}/workspace_mask.png"

echo "[6/9] Run GraspNet on the segmented target cloud"
(
  cd "${GRASPNET_DIR}"
  "${HOST_PYTHON}" "${GRASPNET_RUNNER}" \
    --checkpoint_path "${GRASPNET_CHECKPOINT}" \
    --data_dir "${INPUT_HOST}" --no_vis \
    --save_grasps "${RESULT_HOST}/graspnet_target_grasps.npy"
)

echo "[7/9] Select front-approach, camera-side, midbody grasps"
"${HOST_PYTHON}" "${REPO_ROOT}/scripts/perception/select_grasp.py" \
  --grasps "${RESULT_HOST}/graspnet_target_grasps.npy" \
  --meta "${INPUT_HOST}/meta.mat" --image "${INPUT_HOST}/color.png" \
  --depth "${INPUT_HOST}/depth.png" --mask "${RESULT_HOST}/sam_mask_selected_raw.png" \
  --summary "${RESULT_HOST}/yoloe_sam_selected_summary.json" \
  --out-dir "${RESULT_HOST}" --min-bbox-y 0.30 --max-bbox-y 0.70 \
  --min-mask-world-z-ratio 0.45 --max-mapped-approach-y 0.20 \
  --max-front-approach-angle-deg 30 --pregrasp-distance 0.08 \
  >"${RESULT_HOST}/front_selection.log" 2>&1

cp "${RESULT_HOST}/graspnet_target_top_grasp_midbody.json" \
  "${WORK_HOST}/selected_grasp.json"

echo "[8/9] Solve pregrasp, grasp and retract IK"
docker exec "${ISAAC_CONTAINER}" bash -lc \
  "/isaac-sim/python.sh '${REPO_ISAAC}/scripts/planning/solve_ik.py' \
    --usd '${USD_ISAAC}' --grasp-json '${GRASP_JSON_ISAAC}' \
    --tcp-extrinsic-json '${TCP_EXTRINSIC_ISAAC}' \
    --graspnet-to-ee-mode right --grasp-approach-axis 2 --grasp-opening-axis 0 \
    --output '${IK_ISAAC}' --base '${BASE_X}' '${BASE_Y}' '${BASE_Z}' '${BASE_YAW}' \
    --lift '${LIFT_POSITION}' --max-iters 300 --solver least_squares \
    --position-weight 35 --approach-axis-weight 8 --opening-axis-weight 2 \
    --pregrasp-distance 0.08 --retract-distance 0.10 --retract-lift 0.05 \
    --ee-reference-joints 0 0 0 0 0 0 --max-ee-angle-deg 60 \
    --search-base-lift --base-x-offsets=-0.20,-0.10,0 \
    --base-y-offsets=-0.05,0,0.05 --lift-offsets=-0.05,0,0.05 \
    --search-max-iters 45 > /tmp/left_arm_front_ik.log 2>&1"

docker cp "${ISAAC_CONTAINER}:${IK_ISAAC}" "${RESULT_HOST}/ik.json" >/dev/null
docker cp "${ISAAC_CONTAINER}:/tmp/left_arm_front_ik.log" "${RESULT_HOST}/ik.log" >/dev/null

"${HOST_PYTHON}" "${REPO_ROOT}/scripts/planning/visualize_plan.py" \
  --image "${CAPTURE_HOST}/color.png" --depth "${CAPTURE_HOST}/depth.png" \
  --mask "${RESULT_HOST}/sam_mask_selected_raw.png" --meta "${CAPTURE_HOST}/meta.mat" \
  --detection "${RESULT_HOST}/yoloe_detection_selected.png" \
  --grasp "${RESULT_HOST}/graspnet_target_top_grasp_midbody.json" \
  --ik "${RESULT_HOST}/ik.json" --output "${RESULT_HOST}/grasp_preview.png" \
  --metrics "${RESULT_HOST}/metrics.json"

"${HOST_PYTHON}" - "${RESULT_HOST}/ik.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
stages = data.get("trajectory_converged", {})
if not data.get("converged") or not all(stages.get(k) for k in ("pregrasp", "grasp", "retract")):
    raise SystemExit("IK validation failed; refusing execution")
print(f"IK PASS: position={data['distance_m'] * 1000:.2f} mm, EE angle={data['ee_angle_from_reference_deg']:.2f} deg")
PY

echo "[9/9] Preview saved to ${RESULT_HOST}/grasp_preview.png"
if [[ "${MODE}" == "preview" ]]; then
  exit 0
fi

echo "Executing the validated trajectory"
docker cp "${RESULT_HOST}/ik.json" "${ROS_CONTAINER}:/tmp/left_arm_graspnet_ik.json" >/dev/null
docker cp "${REPO_ROOT}/scripts/ros/execute_grasp.py" "${ROS_CONTAINER}:${ROS_EXECUTOR}" >/dev/null
ros_exec "python3 '${ROS_EXECUTOR}'" 2>&1 | tee "${RESULT_HOST}/execution.log"
