"""EXACT ManiSkill PushCube-v1 style dense reward.

Uses 1 - tanh() bounded rewards and reached masking to prevent Value Loss explosion.
Cube size: (0.1, 0.1, 0.2). Half-size X = 0.05. Initial center Z = 0.15, bottom Z = 0.05.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv


def ms_reaching_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Pose marking behind the cube. Bounded tanh reward [0, 1]."""
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    obj_pos = env.scene["object"].data.root_pos_w

    tcp_push_pose = obj_pos.clone()
    tcp_push_pose[:, 0] -= (0.05 + 0.005)  # -0.055 (Behind the cube)
    tcp_push_pose[:, 2] -= 0.05  # Lower part of the cube to prevent tipping

    tcp_to_push_pose_dist = torch.norm(tcp_push_pose - ee_pos, p=2, dim=-1)

    return 1.0 - torch.tanh(5.0 * tcp_to_push_pose_dist)


def ms_push_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Push reward with reached mask. Only reward pushing when EE is in position."""
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    obj_pos = env.scene["object"].data.root_pos_w

    tcp_push_pose = obj_pos.clone()
    tcp_push_pose[:, 0] -= 0.055
    tcp_push_pose[:, 2] -= 0.05
    tcp_to_push_pose_dist = torch.norm(tcp_push_pose - ee_pos, p=2, dim=-1)

    reached = tcp_to_push_pose_dist < 0.05  # Mask

    obj_vel_x = env.scene["object"].data.root_lin_vel_w[:, 0]
    push_reward = torch.tanh(3.0 * torch.clamp(obj_vel_x, min=0.0))

    return push_reward * reached.float()


def ms_z_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Z-stability reward: only active when reached and pushing. Penalizes Z lift."""
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    obj_pos = env.scene["object"].data.root_pos_w

    tcp_push_pose = obj_pos.clone()
    tcp_push_pose[:, 0] -= 0.055
    tcp_push_pose[:, 2] -= 0.05
    tcp_to_push_pose_dist = torch.norm(tcp_push_pose - ee_pos, p=2, dim=-1)
    reached = tcp_to_push_pose_dist < 0.05

    obj_vel_x = env.scene["object"].data.root_lin_vel_w[:, 0]
    push_reward = torch.tanh(3.0 * torch.clamp(obj_vel_x, min=0.0))

    desired_obj_z = 0.15  # Initial center Z
    current_obj_z = obj_pos[:, 2]
    z_deviation = torch.abs(current_obj_z - desired_obj_z)
    z_reward = 1.0 - torch.tanh(10.0 * z_deviation)

    return push_reward * z_reward * reached.float()


def ms_y_drift_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Keep the cube strictly on the X-axis rail. Returns [0, 1]."""
    obj_y = env.scene["object"].data.root_pos_w[:, 1]
    return 1.0 - torch.tanh(10.0 * torch.abs(obj_y))
