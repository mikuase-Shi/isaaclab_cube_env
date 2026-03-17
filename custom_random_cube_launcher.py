import sys
import os
import runpy


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# IsaacLab root where rsl_rl scripts live
isaaclab_root = "/root/IsaacLab"

os.chdir(isaaclab_root)

# Ensure the environment package is imported so that its gym envs are registered
import push_random_cube  # noqa: F401

# Add rsl_rl training scripts to path
sys.path.append(os.path.join(isaaclab_root, "scripts/reinforcement_learning/rsl_rl"))

# Configure training arguments for the orientation-aware task
sys.argv = [
    "train.py",
    "--task=Isaac-Push-Random-Franka-v0",
    "--num_envs=4096",
    "--headless",
]

# Launch RSL-RL PPO training
runpy.run_path(
    os.path.join(isaaclab_root, "scripts/reinforcement_learning/rsl_rl/train.py"),
    run_name="__main__",
)
