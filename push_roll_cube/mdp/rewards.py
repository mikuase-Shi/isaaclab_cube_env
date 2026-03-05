import torch
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils

def object_local_z_alignment(env: ManagerBasedRLEnv) -> torch.Tensor:
    quat = env.scene["object"].data.root_quat_w
    base_z = torch.zeros((quat.shape[0], 3), device=quat.device)
    base_z[:, 2] = 1.0
    
    world_z_of_object = math_utils.quat_apply(quat, base_z)
    
    return torch.abs(world_z_of_object[:, 2])

def object_x_displacement(env: ManagerBasedRLEnv) -> torch.Tensor:
    object_pos = env.scene["object"].data.root_pos_w
    
    return (object_pos[:, 0] - 0.5)