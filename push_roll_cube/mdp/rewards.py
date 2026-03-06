import torch
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils

def object_local_z_alignment(env: ManagerBasedRLEnv) -> torch.Tensor:
    quat = env.scene["object"].data.root_quat_w
    base_z = torch.zeros((quat.shape[0], 3), device=quat.device)
    base_z[:, 2] = 1.0
    
    world_z_of_object = math_utils.quat_apply(quat, base_z)
    
    return torch.abs(world_z_of_object[:, 2])


def object_x_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Reward the linear velocity of the object along the world X-axis
    object_lin_vel = env.scene["object"].data.root_lin_vel_w
    return object_lin_vel[:, 0]

def object_angular_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Reward the magnitude of angular velocity to encourage continuous rolling
    object_ang_vel = env.scene["object"].data.root_ang_vel_w
    return torch.norm(object_ang_vel, p=2, dim=-1)

def object_x_displacement(env: ManagerBasedRLEnv) -> torch.Tensor:
    object_pos = env.scene["object"].data.root_pos_w
    
    return (object_pos[:, 0] - 0.5)

def ee_object_distance_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_pos=env.scene["ee_frame"].data.target_pos_w[...,0,:]
    object_pos=env.scene["object"].data.root_pos_w
    distance = torch.norm(ee_pos - object_pos, p=2, dim=-1)
    return distance