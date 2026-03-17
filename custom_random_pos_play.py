import sys
import os
import argparse
import torch

# ── Path setup ──────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

isaaclab_root = "/root/IsaacLab"
os.chdir(isaaclab_root)
sys.path.append(os.path.join(isaaclab_root, "scripts/reinforcement_learning/rsl_rl"))

# 🚨 关键点：必须在导入任何底层包前，先启动 AppLauncher
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnostic play script")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pt)")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps to run")
parser.add_argument("--env_id", type=int, default=0, help="Which env index to print diagnostics for")
parser.add_argument("--print_every", type=int, default=10, help="Print diagnostics every N steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动物理引擎
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 引擎启动后，再去 import 环境和 Reward 函数
import gymnasium as gym
import push_random_pos_cube  # register the env

from push_random_pos_cube.mdp.rewards import (
    _get_tcp_push_pose,
    _get_dist_to_goal_2d,
    _get_reach_multiplier,
    ms_phased_goal_reward,
    ms_stationary_reward,
    ms_fine_position_reward,
    ms_near_goal_vel_penalty,
    ms_overshoot_penalty,
)

def load_policy(checkpoint_path, device):
    from rsl_rl.modules import ActorCritic
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = loaded.get("model_state_dict", loaded)

    actor_first_key = [k for k in state_dict if k.startswith("actor.")][0]
    actor_last_key  = [k for k in state_dict if k.startswith("actor.")][-1]
    obs_dim = state_dict[actor_first_key].shape[1] 
    act_dim = state_dict[actor_last_key].shape[0]  

    model = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=act_dim,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def print_header():
    print("\n" + "=" * 120)
    print(f"{'Step':>5} | {'Obj X':>7} {'Obj Y':>7} {'Obj Z':>7} | "
          f"{'Goal X':>7} {'Goal Y':>7} | {'Dist2D':>7} | "
          f"{'ObjVx':>7} {'ObjVy':>7} {'Speed':>7} | "
          f"{'EE X':>7} {'EE Y':>7} {'EE Z':>7} | "
          f"{'PP Dst':>7} {'Reach':>6} | "
          f"{'PhasR':>6} {'StatR':>6} {'FineR':>6} {'VelP':>6} {'OvrP':>6}")
    print("-" * 120)

def print_diagnostics(env, step_idx, eid):
    i = eid
    obj_pos  = env.unwrapped.scene["object"].data.root_pos_w[i].cpu()
    goal_pos = env.unwrapped.scene["goal"].data.root_pos_w[i].cpu()
    ee_pos   = env.unwrapped.scene["ee_frame"].data.target_pos_w[i, 0, :].cpu()

    obj_lin_vel = env.unwrapped.scene["object"].data.root_lin_vel_w[i].cpu()
    speed_xy = torch.norm(obj_lin_vel[:2], p=2).item()
    dist_2d = torch.norm(goal_pos[:2] - obj_pos[:2], p=2).item()

    push_pose = _get_tcp_push_pose(env.unwrapped)[i].cpu()
    pp_dist = torch.norm(push_pose - ee_pos, p=2).item()
    reach_mult = _get_reach_multiplier(env.unwrapped)[i].item()

    phased  = ms_phased_goal_reward(env.unwrapped)[i].item()
    stat    = ms_stationary_reward(env.unwrapped)[i].item()
    fine    = ms_fine_position_reward(env.unwrapped)[i].item()
    vel_pen = ms_near_goal_vel_penalty(env.unwrapped)[i].item()
    ovr_pen = ms_overshoot_penalty(env.unwrapped)[i].item()

    print(f"{step_idx:5d} | "
          f"{obj_pos[0].item():7.3f} {obj_pos[1].item():7.3f} {obj_pos[2].item():7.3f} | "
          f"{goal_pos[0].item():7.3f} {goal_pos[1].item():7.3f} | "
          f"{dist_2d:7.4f} | "
          f"{obj_lin_vel[0].item():7.3f} {obj_lin_vel[1].item():7.3f} {speed_xy:7.4f} | "
          f"{ee_pos[0].item():7.3f} {ee_pos[1].item():7.3f} {ee_pos[2].item():7.3f} | "
          f"{pp_dist:7.4f} {reach_mult:6.3f} | "
          f"{phased:6.3f} {stat:6.3f} {fine:6.3f} {vel_pen:6.3f} {ovr_pen:6.3f}")

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"📦 Loading env: Isaac-Push-Random-Pos-Franka-v0 ({args_cli.num_envs} envs)")
    
    # 🚨 关键修复：导入 IsaacLab 的配置解析工具，先生成 cfg，再喂给 gym.make
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    env_cfg = parse_env_cfg("Isaac-Push-Random-Pos-Franka-v0", device=device, num_envs=args_cli.num_envs)
    
    # 将解析好的 cfg 传递给 ManagerBasedRLEnv
    env = gym.make("Isaac-Push-Random-Pos-Franka-v0", cfg=env_cfg)

    print(f"🧠 Loading checkpoint: {args_cli.checkpoint}")
    model = load_policy(args_cli.checkpoint, device)

    print(f"🔍 Printing diagnostics for env index {args_cli.env_id}, every {args_cli.print_every} steps")
    print_header()

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    for step in range(args_cli.max_steps):
        with torch.no_grad():
            actions = model.act_inference(obs)

        obs_dict, rewards, dones, truncated, info = env.step(actions)
        obs = obs_dict["policy"]

        if step % args_cli.print_every == 0:
            print_diagnostics(env, step, args_cli.env_id)

        if dones[args_cli.env_id] or truncated[args_cli.env_id]:
            print(f"  >>> ENV {args_cli.env_id} RESET at step {step} (done={dones[args_cli.env_id].item()}, trunc={truncated[args_cli.env_id].item()})")
            print_header()

    print("\n✅ Done!")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()