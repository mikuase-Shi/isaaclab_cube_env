import sys
import os
import runpy

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

isaaclab_root = "/root/IsaacLab" 

os.chdir(isaaclab_root)

try:
    import push_roll_cube
    print("✅ 环境身份证注册成功！")
except ImportError as e:
    print(f"❌ 注册失败：{e}")
    sys.exit(1)

sys.path.append(os.path.join(isaaclab_root, "scripts/reinforcement_learning/rsl_rl"))

sys.argv = [
    "train.py", 
    "--task=Isaac-Push-Flip-Franka-v0", 
    "--num_envs=64", 
    "--headless",
]

runpy.run_path(os.path.join(isaaclab_root, "scripts/reinforcement_learning/rsl_rl/train.py"), run_name="__main__")