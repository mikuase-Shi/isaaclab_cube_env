import sys
import os
import runpy

sys.path.append("/root/gpufree-data/isaaclab_cube_env")

import push_roll_cube

sys.path.append(os.path.abspath("scripts/reinforcement_learning/rsl_rl"))

sys.argv = [
    "train.py", 
    "--task=Isaac-Push-Flip-Franka-v0", 
    "--num_envs=4096", 
    "--headless"
]

runpy.run_path("scripts/reinforcement_learning/rsl_rl/train.py", run_name="__main__")