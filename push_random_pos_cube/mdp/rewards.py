"""Rewards for random-position cube pushing with stable gradients.

- Dynamic push pose: aligned with object→goal direction.
- Position-only goal shaping (XY), no orientation terms.
- Far distances use linear shaping; near goal use tanh for smoothness.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv


def _get_tcp_push_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compute push pose dynamically along the object→goal direction.

    - Direction is from object to goal in XY.
    - We go 0.055m *behind* the cube along the negative direction.
    - Z is set to lower part of the cube (center z - 0.05).
    """
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w

    # Vector from object to goal in XY
    dir_xy = goal_pos[:, :2] - obj_pos[:, :2]
    dist_xy = torch.norm(dir_xy, p=2, dim=-1, keepdim=True)

    # Fallback: if goal almost coincides with object, use -X direction
    eps = 1e-6
    default_dir = torch.tensor([-1.0, 0.0], device=obj_pos.device).view(1, 2)
    dir_xy_unit = torch.where(
        dist_xy > eps,
        dir_xy / (dist_xy + 1e-8),
        default_dir.expand_as(dir_xy),
    )

    tcp_push_pose = obj_pos.clone()
    # Move 5.5 cm behind the cube along -dir_xy_unit
    tcp_push_pose[:, :2] = obj_pos[:, :2] - 0.055 * dir_xy_unit
    # Lower part of cube to reduce tipping
    tcp_push_pose[:, 2] = obj_pos[:, 2] - 0.05
    return tcp_push_pose


def _get_reach_multiplier(env: ManagerBasedRLEnv) -> torch.Tensor:
    """EE close to push pose: 1 - tanh(5 * dist).

    Softer scaling than 10x to avoid vanishing gradient.
    """
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    tcp_push_pose = _get_tcp_push_pose(env)
    tcp_to_push_pose_dist = torch.norm(tcp_push_pose - ee_pos, p=2, dim=-1)
    return 1.0 - torch.tanh(5.0 * tcp_to_push_pose_dist)


def _get_dist_to_goal_2d(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w
    return torch.norm(goal_pos[:, :2] - obj_pos[:, :2], p=2, dim=-1)


def ms_reaching_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """EE to dynamic push pose behind cube. Bounded [0, 1]."""
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    tcp_push_pose = _get_tcp_push_pose(env)
    tcp_to_push_pose_dist = torch.norm(tcp_push_pose - ee_pos, p=2, dim=-1)
    return 1.0 - torch.tanh(5.0 * tcp_to_push_pose_dist)


def ms_goal_reaching_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Main goal position reward with far/near shaping to avoid vanishing gradients."""
    reach_multiplier = _get_reach_multiplier(env)
    dist_to_goal_2d = _get_dist_to_goal_2d(env)

    # Far: linear shaping; Near (<0.1m): tanh for smooth convergence
    far_mask = dist_to_goal_2d >= 0.1
    near_mask = ~far_mask

    # Far: 1 - dist / 0.5 (clipped to [0,1])
    far_reward = 1.0 - torch.clamp(dist_to_goal_2d / 0.5, min=0.0, max=1.0)

    # Near: 1 - tanh(2 * dist)
    near_reward = 1.0 - torch.tanh(2.0 * dist_to_goal_2d)

    push_reward = torch.where(far_mask, far_reward, near_reward)
    return push_reward * reach_multiplier


def ms_fine_position_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fine-tuning: only when dist_2d < 0.03, strong gradient for last centimeters."""
    dist_to_goal_2d = _get_dist_to_goal_2d(env)
    fine_reward = 1.0 - torch.tanh(10.0 * dist_to_goal_2d)
    close_mask = (dist_to_goal_2d < 0.03).float()
    return fine_reward * close_mask


def ms_goal_pos_x_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-axis: reward for X alignment with goal (position only)."""
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w
    dx = torch.abs(goal_pos[:, 0] - obj_pos[:, 0])
    return 1.0 - torch.tanh(5.0 * dx)


def ms_goal_pos_y_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-axis: reward for Y alignment with goal (position only)."""
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w
    dy = torch.abs(goal_pos[:, 1] - obj_pos[:, 1])
    return 1.0 - torch.tanh(5.0 * dy)


def ms_near_goal_vel_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize object XY velocity when close to goal to encourage settling."""
    dist_to_goal_2d = _get_dist_to_goal_2d(env)
    obj_lin_vel = env.scene["object"].data.root_lin_vel_w
    vel_xy = torch.norm(obj_lin_vel[:, :2], p=2, dim=-1)
    close_mask = (dist_to_goal_2d < 0.05).float()
    return vel_xy * close_mask  # positive value -> use negative weight in cfg


def ms_z_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Z-stability when reached and pushing (no rotation requirements)."""
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    obj_pos = env.scene["object"].data.root_pos_w
    tcp_push_pose = _get_tcp_push_pose(env)
    tcp_to_push_pose_dist = torch.norm(tcp_push_pose - ee_pos, p=2, dim=-1)
    reached = tcp_to_push_pose_dist < 0.05

    obj_vel_x = env.scene["object"].data.root_lin_vel_w[:, 0]
    push_reward = torch.tanh(3.0 * torch.clamp(obj_vel_x, min=0.0))

    desired_obj_z = 0.15
    current_obj_z = obj_pos[:, 2]
    z_deviation = torch.abs(current_obj_z - desired_obj_z)
    z_reward = 1.0 - torch.tanh(10.0 * z_deviation)

    return push_reward * z_reward * reached.float()
