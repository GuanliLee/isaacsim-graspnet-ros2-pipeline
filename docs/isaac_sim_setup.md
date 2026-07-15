# Isaac Sim and Container Setup

## Required Mounts

The pipeline runner executes two Isaac-side Python programs. Mount this private
repository into the Isaac Sim container; the portable USD bundle is included:

```bash
-v /absolute/path/isaacsim-graspnet-ros2-pipeline:/workspace/grasp-pipeline
```

Use host networking for ROS 2 DDS discovery when both containers run on the
same machine:

```bash
--network host
```

Set the same `ROS_DOMAIN_ID` in `config/pipeline.env`, the ROS 2 container, and
the Isaac Sim Script Node environment.

## OmniGraph Script Node

1. Open the target USD in Isaac Sim.
2. Add or select an OmniGraph Script Node.
3. Enable `Use Path`.
4. Set the path to:

```text
/workspace/grasp-pipeline/scripts/ros/isaac_topic_control.py
```

5. Press Reload Script, then Play.

Verify from the ROS 2 container:

```bash
export ROS_DOMAIN_ID=44
source /opt/ros/jazzy/setup.bash
ros2 topic info /joint_states_gripper_l -v
ros2 topic echo /left_arm/current_joint_states --once
```

## Large Private Assets

The `model/` and `assets/usd/` trees are committed through Git LFS. Run
`git lfs pull` after cloning. Captured RGB-D frames, inference results, Python
build outputs, and machine-specific `pipeline.env` remain ignored.
