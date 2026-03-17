"""Rewards for random-position cube pushing with stable gradients.

- Dynamic push pose: aligned with object→goal direction.
- Position-only goal shaping (XY), no orientation terms.
- Phase-aware goal reward: different shaping per distance phase (far/mid/near).
- Stationary reward: encourages object to stop when close to goal.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv


def _get_tcp_push_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w

    dir_xy = goal_pos[:, :2] - obj_pos[:, :2]
    dist_xy = torch.norm(dir_xy, p=2, dim=-1, keepdim=True)

    eps = 1e-6
    default_dir = torch.tensor([-1.0, 0.0], device=obj_pos.device).view(1, 2)
    dir_xy_unit = torch.where(
        dist_xy > eps,
        dir_xy / (dist_xy + 1e-8),
        default_dir.expand_as(dir_xy),
    )

    tcp_push_pose = obj_pos.clone()
    tcp_push_pose[:, :2] = obj_pos[:, :2] - 0.055 * dir_xy_unit
    tcp_push_pose[:, 2] = obj_pos[:, 2] - 0.05
    return tcp_push_pose


def _get_reach_multiplier(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    tcp_push_pose = _get_tcp_push_pose(env)
    tcp_to_push_pose_dist = torch.norm(tcp_push_pose - ee_pos, p=2, dim=-1)
    return 1.0 - torch.tanh(5.0 * tcp_to_push_pose_dist)


def _get_dist_to_goal_2d(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w
    return torch.norm(goal_pos[:, :2] - obj_pos[:, :2], p=2, dim=-1)


def ms_reaching_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    tcp_push_pose = _get_tcp_push_pose(env)
    tcp_to_push_pose_dist = torch.norm(tcp_push_pose - ee_pos, p=2, dim=-1)
    return 1.0 - torch.tanh(5.0 * tcp_to_push_pose_dist)


def ms_phased_goal_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Unified goal reward — no phase boundaries.

    R = (1 - tanh(3 * dist)) * reach_multiplier
    Smooth everywhere, no distance gates, no dead zones.
    """
    reach_multiplier = _get_reach_multiplier(env)
    dist = _get_dist_to_goal_2d(env)
    return (1.0 - torch.tanh(3.0 * dist)) * reach_multiplier


def ms_stationary_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward the object for being stationary when close to the goal.

    R = exp(-3 * speed_xy), peaks at 1 when stopped, decays as object moves.
    Gated by two distance rings so the reward only fires near the goal:
      - Within 0.08m: base gate (weight 1x)
      - Within 0.04m: extra gate (another 1x, total 2x bonus for being close AND still)
    """
    dist = _get_dist_to_goal_2d(env)
    speed_xy = torch.norm(env.scene["object"].data.root_lin_vel_w[:, :2], p=2, dim=-1)

    gate_mid  = (dist < 0.08).float()
    gate_near = (dist < 0.04).float()
    stop_reward = torch.exp(-3.0 * speed_xy)
    return stop_reward * (gate_mid + gate_near)




def ms_goal_pos_x_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w
    dx = torch.abs(goal_pos[:, 0] - obj_pos[:, 0])
    return 1.0 - torch.tanh(5.0 * dx)


def ms_goal_pos_y_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w
    dy = torch.abs(goal_pos[:, 1] - obj_pos[:, 1])
    return 1.0 - torch.tanh(5.0 * dy)


def ms_near_goal_vel_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    dist_to_goal_2d = _get_dist_to_goal_2d(env)
    obj_lin_vel = env.scene["object"].data.root_lin_vel_w
    vel_xy = torch.norm(obj_lin_vel[:, :2], p=2, dim=-1)
    
    close_mask = (dist_to_goal_2d < 0.05).float()
    very_close_mask = (dist_to_goal_2d < 0.02).float()
    
    return vel_xy * (close_mask + very_close_mask)


def ms_overshoot_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w
    obj_lin_vel = env.scene["object"].data.root_lin_vel_w
    dist_to_goal_2d = _get_dist_to_goal_2d(env)

    to_obj = obj_pos[:, :2] - goal_pos[:, :2]
    to_obj_norm = torch.norm(to_obj, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
    to_obj_unit = to_obj / to_obj_norm

    vel_away = (obj_lin_vel[:, :2] * to_obj_unit).sum(dim=-1).clamp(min=0.0)

    close_mask = (dist_to_goal_2d < 0.06).float()
    return vel_away * close_mask


def ms_past_goal_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the object for overshooting past the goal along the start→goal axis.

    The object start position (0.5, 0.0) in local frame is converted to world
    coordinates via env_origins so this works correctly in multi-env setups.

    How it works:
      1. start_w = env_origins[:, :2] + (0.5, 0.0)
      2. Compute unit vector  d = normalize(goal - start)
      3. Project object position onto this axis:
           proj = dot(obj - start, d)
           goal_proj = dot(goal - start, d)
      4. overshoot = max(0, proj - goal_proj)   (positive = past the goal)
      5. penalty = tanh(5 * overshoot)           (smooth, bounded 0-1)
    """
    obj_pos  = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w
    origins  = env.scene.env_origins

    # Object start position in world frame
    start_w = origins[:, :2].clone()
    start_w[:, 0] += 0.5  # local x offset of the object init_state
    # start_w[:, 1] += 0.0  # local y offset is 0

    # Start → goal direction (2D)
    sg = goal_pos[:, :2] - start_w
    sg_len = torch.norm(sg, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
    sg_unit = sg / sg_len

    # Project object and goal onto start→goal axis
    obj_proj  = ((obj_pos[:, :2] - start_w) * sg_unit).sum(dim=-1)
    goal_proj = sg_len.squeeze(-1)  # = dot(goal - start, sg_unit) = |sg|

    # Overshoot: how far past the goal the object has gone
    overshoot = (obj_proj - goal_proj).clamp(min=0.0)

    return torch.tanh(5.0 * overshoot)


def ms_z_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj_pos = env.scene["object"].data.root_pos_w
    
    current_obj_z = obj_pos[:, 2]
    z_deviation = torch.abs(current_obj_z - 0.15) 
    z_reward = 1.0 - torch.tanh(10.0 * z_deviation)

    return z_reward
