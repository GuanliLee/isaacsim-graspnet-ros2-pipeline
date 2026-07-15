# Dependency Boundaries

This private repository contains pipeline-specific integration code plus the
local third-party source snapshots used by the tested environment.

| Function | Pipeline code in this repository | Required external implementation |
|---|---|---|
| YOLOE detection | `scripts/perception/detect_target.py` | `third_party/ultralytics` and `model/yoloe` |
| SAM segmentation | `scripts/perception/segment_target.py` | `third_party/ultralytics` and `model/sam` |
| Grasp generation | `run_pipeline.sh` integration | `third_party/graspnet-baseline` and `model/graspnet` |
| Grasp representation/filtering | `scripts/perception/select_grasp.py` | `third_party/graspnetAPI` |
| TCP-aware IK | `scripts/planning/solve_ik.py` | Isaac Sim Python runtime |
| ROS execution | `scripts/ros/execute_grasp.py` | ROS 2 Jazzy and the Isaac Script Node bridge |

## GraspNetAPI Is Not the Grasp Network

GraspNetAPI provides grasp data structures, NMS, visualization, dataset access,
and evaluation helpers. `GraspGroup` loads the `.npy` candidates and exposes
translations, rotations, widths, and scores. It does not contain the neural
network that predicts those candidates from an RGB-D point cloud.

GraspNet Baseline provides the actual PyTorch network, point-cloud backbone,
PointNet2/KNN extensions, decoding, collision checking, and checkpoint loader.
Both are needed for the complete pipeline and are included under
`third_party/` with their original licenses.

## Private Source Snapshots

- Source snapshots retain their original license files.
- Binary models and USD assets are managed by Git LFS.
- The root MIT license applies only to original integration code.
- Do not make the repository public without removing restricted content.

A fresh machine still needs compatible CUDA, PyTorch, Isaac Sim, ROS 2, and
compiled PointNet2/KNN extensions. Committing source does not package those
system runtimes.

## Local GraspNetAPI Checkout

To use a local GraspNetAPI checkout instead of the PyPI package:

```bash
python -m pip uninstall -y graspnetAPI
python -m pip install -e /absolute/path/to/graspnetAPI
```

Confirm the imported location:

```bash
python -c "import graspnetAPI; print(graspnetAPI.__file__)"
```
