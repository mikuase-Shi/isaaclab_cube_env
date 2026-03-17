import sys
import os
import runpy

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

isaaclab_root = "/root/IsaacLab" 
os.chdir(isaaclab_root)

# 强行导入修复版环境包
try:
    import push_random_pos_cube
    print("✅ 成功导入修复版环境：push_random_pos_cube")
except ImportError as e:
    print(f"❌ 导入环境失败：{e}")
    sys.exit(1)

sys.path.append(os.path.join(isaaclab_root, "scripts/reinforcement_learning/rsl_rl"))

# 配置 play 脚本参数
sys.argv = [
    "play.py", 
    "--task=Isaac-Push-Random-Pos-Franka-v0",  
    "--num_envs=32",        # 环境数量，录制视频时建议调小一些
    "--headless",           # 🚨 隐藏渲染窗口 (不弹窗)
    "--video",              # 🚨 开启视频录制功能
    "--video_length=400",   # 🚨 录制的物理步数 (400步通常对应十几秒，可自行调整)
]

print("🎥 正在后台启动推理并录制视频 (Headless Video Recording)...")
runpy.run_path(os.path.join(isaaclab_root, "scripts/reinforcement_learning/rsl_rl/play.py"), run_name="__main__")