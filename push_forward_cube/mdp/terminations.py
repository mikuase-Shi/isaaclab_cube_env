import torch
from isaaclab.envs import ManagerBasedRLEnv

def object_moved_x_cm(env: ManagerBasedRLEnv, distance: float) -> torch.Tensor:
    object_pos_w = env.scene["object"].data.root_pos_w
    env_origins = env.scene.env_origins
    
    object_pos_local_x = object_pos_w[:, 0] - env_origins[:, 0]
    
    return (object_pos_local_x - 0.5) >= distance