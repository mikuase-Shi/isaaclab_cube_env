"""Goal-reaching rewards with fine-grained position/alignment for near-exact overlap."""

import torch
from isaaclab.envs import ManagerBasedRLEnv


def _get_tcp_push_pose(obj_pos: torch.Tensor) -> torch.Tensor:
    """Target pose behind cube: -0.055 in X, -0.05 in Z."""
    tcp_push_pose = obj_pos.clone()
    tcp_push_pose[:, 0] -= 0.055
    tcp_push_pose[:, 2] -= 0.05
    return tcp_push_pose


def _get_reach_multiplier(env: ManagerBasedRLEnv) -> torch.Tensor:
    """EE close to push pose: 1 - tanh(10 * dist)."""
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    obj_pos = env.scene["object"].data.root_pos_w
    tcp_push_pose = _get_tcp_push_pose(obj_pos)
    tcp_to_push_pose_dist = torch.norm(tcp_push_pose - ee_pos, p=2, dim=-1)
    return 1.0 - torch.tanh(10.0 * tcp_to_push_pose_dist)


def _get_dist_to_goal_2d(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w
    return torch.norm(goal_pos[:, :2] - obj_pos[:, :2], p=2, dim=-1)


def _get_angle_diff_to_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    import isaaclab.utils.math as math_utils
    obj_quat = env.scene["object"].data.root_quat_w
    goal_quat = env.scene["goal"].data.root_quat_w
    obj_quat_inv = math_utils.quat_inv(obj_quat)
    quat_diff = math_utils.quat_mul(obj_quat_inv, goal_quat)
    angle_diff = 2.0 * torch.acos(torch.clamp(quat_diff[:, 0], -1.0, 1.0))
    angle_diff = torch.where(angle_diff > torch.pi, 2 * torch.pi - angle_diff, angle_diff)
    return angle_diff


def ms_reaching_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """EE to push pose behind cube. Bounded [0, 1]."""
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    obj_pos = env.scene["object"].data.root_pos_w
    tcp_push_pose = _get_tcp_push_pose(obj_pos)
    tcp_to_push_pose_dist = torch.norm(tcp_push_pose - ee_pos, p=2, dim=-1)
    return 1.0 - torch.tanh(5.0 * tcp_to_push_pose_dist)


def ms_goal_reaching_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Main goal position reward. Higher sensitivity: tanh(10 * dist_2d)."""
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w
    reach_multiplier = _get_reach_multiplier(env)
    dist_to_goal_2d = torch.norm(goal_pos[:, :2] - obj_pos[:, :2], p=2, dim=-1)
    push_reward = 1.0 - torch.tanh(10.0 * dist_to_goal_2d)
    return push_reward * reach_multiplier


def ms_fine_position_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fine-tuning: only when dist_2d < 0.03, strong gradient for last 1--2 cm."""
    dist_to_goal_2d = _get_dist_to_goal_2d(env)
    fine_reward = 1.0 - torch.tanh(30.0 * dist_to_goal_2d)
    close_mask = (dist_to_goal_2d < 0.03).float()
    return fine_reward * close_mask


def ms_goal_pos_x_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-axis: reward for X alignment with goal."""
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w
    dx = torch.abs(goal_pos[:, 0] - obj_pos[:, 0])
    return 1.0 - torch.tanh(15.0 * dx)


def ms_goal_pos_y_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-axis: reward for Y alignment with goal."""
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w
    dy = torch.abs(goal_pos[:, 1] - obj_pos[:, 1])
    return 1.0 - torch.tanh(15.0 * dy)


def ms_goal_alignment_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Orientation alignment; only emphasized when within 5 cm of goal."""
    dist_to_goal_2d = _get_dist_to_goal_2d(env)
    close_to_goal_multiplier = torch.clamp(1.0 - dist_to_goal_2d / 0.05, 0.0, 1.0)
    angle_diff = _get_angle_diff_to_goal(env)
    alignment_reward = 1.0 - torch.tanh(8.0 * angle_diff)
    return alignment_reward * close_to_goal_multiplier


def ms_fine_alignment_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fine angle tuning when already very close (dist < 2 cm)."""
    dist_to_goal_2d = _get_dist_to_goal_2d(env)
    angle_diff = _get_angle_diff_to_goal(env)
    fine_align = 1.0 - torch.tanh(20.0 * angle_diff)
    close_mask = (dist_to_goal_2d < 0.02).float()
    return fine_align * close_mask


def ms_near_goal_vel_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize object XY velocity when close to goal to encourage settling."""
    dist_to_goal_2d = _get_dist_to_goal_2d(env)
    obj_lin_vel = env.scene["object"].data.root_lin_vel_w
    vel_xy = torch.norm(obj_lin_vel[:, :2], p=2, dim=-1)
    close_mask = (dist_to_goal_2d < 0.05).float()
    return vel_xy * close_mask  # positive value -> use negative weight in cfg


def ms_z_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Z-stability when reached and pushing."""
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    obj_pos = env.scene["object"].data.root_pos_w
    tcp_push_pose = _get_tcp_push_pose(obj_pos)
    tcp_to_push_pose_dist = torch.norm(tcp_push_pose - ee_pos, p=2, dim=-1)
    reached = tcp_to_push_pose_dist < 0.05

    obj_vel_x = env.scene["object"].data.root_lin_vel_w[:, 0]
    push_reward = torch.tanh(3.0 * torch.clamp(obj_vel_x, min=0.0))

    desired_obj_z = 0.15
    current_obj_z = obj_pos[:, 2]
    z_deviation = torch.abs(current_obj_z - desired_obj_z)
    z_reward = 1.0 - torch.tanh(10.0 * z_deviation)

    return push_reward * z_reward * reached.float()
