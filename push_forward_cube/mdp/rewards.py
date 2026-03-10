import torch
import math
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils

def object_local_z_alignment(env: ManagerBasedRLEnv) -> torch.Tensor:
    quat = env.scene["object"].data.root_quat_w
    base_z = torch.zeros((quat.shape[0], 3), device=quat.device)
    base_z[:, 2] = 1.0
    
    world_z_of_object = math_utils.quat_apply(quat, base_z)
    
    return torch.abs(world_z_of_object[:, 2])

def upright_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Strict penalty for deviations of the object's Z-axis from world's Z-axis
    z_alignment = object_local_z_alignment(env)
    return 1.0 - z_alignment

def lateral_drift_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Penalize movement/drift along Y-axis
    object_pos = env.scene["object"].data.root_pos_w
    object_lin_vel = env.scene["object"].data.root_lin_vel_w
    
    # Penalize both Y position deviation (from 0) and Y velocity
    y_drift = torch.square(object_pos[:, 1]) + 0.1 * torch.square(object_lin_vel[:, 1])
    return y_drift

def ee_orientation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Penalize deviation of EE's forward pointing axis (local Z) from world's +X-axis
    quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    base_z = torch.zeros((quat.shape[0], 3), device=quat.device)
    base_z[:, 2] = 1.0  # EE local forward direction
    
    world_forward_of_ee = math_utils.quat_apply(quat, base_z)
    
    # We want world_forward_of_ee to be aligned with (1, 0, 0)
    target_forward = torch.zeros_like(world_forward_of_ee)
    target_forward[:, 0] = 1.0
    
    # Penalize angle difference
    cos_theta = torch.sum(world_forward_of_ee * target_forward, dim=-1)
    return 1.0 - cos_theta

def ee_height_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Target height is around the bottom part of the block (e.g., z = 0.05)
    # The block's center is at 0.15 initially (size z is 0.2, bottom at 0.05)
    ee_z = env.scene["ee_frame"].data.target_pos_w[..., 0, 2]
    target_z = 0.05
    z_dist = torch.square(ee_z - target_z)
    return z_dist

def object_x_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Reward the linear velocity of the object along the world X-axis
    object_lin_vel = env.scene["object"].data.root_lin_vel_w
    return object_lin_vel[:, 0]

def object_x_displacement(env: ManagerBasedRLEnv) -> torch.Tensor:
    object_pos = env.scene["object"].data.root_pos_w
    return (object_pos[:, 0] - 0.5)

def ee_object_distance_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Distance penalty targeting a point slightly behind and below the center of the block
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w
    
    # Target point: -0.06m behind the object along X-axis, and near block's bottom (Z offset: -0.08)
    target_pos = object_pos.clone()
    target_pos[:, 0] -= 0.06
    target_pos[:, 2] -= 0.08
    
    distance = torch.norm(ee_pos - target_pos, p=2, dim=-1)
    return distance