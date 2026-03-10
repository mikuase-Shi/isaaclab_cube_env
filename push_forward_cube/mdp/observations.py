import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

def rel_ee_object_distance(env:ManagerBasedRLEnv)->torch.Tensor:
    ee_pos=env.scene["ee_frame"].data.target_pos_w[...,0,:]
    object_pos=env.scene["object"].data.root_pos_w
    return object_pos-ee_pos

def object_local_z_alignment_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    quat = env.scene["object"].data.root_quat_w
    base_z = torch.zeros((quat.shape[0], 3), device=quat.device)
    base_z[:, 2] = 1.0
    world_z_of_object = math_utils.quat_apply(quat, base_z)
    return torch.abs(world_z_of_object[:, 2]).unsqueeze(-1)

def object_x_displacement_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    object_pos_w = env.scene["object"].data.root_pos_w
    env_origins = env.scene.env_origins

    object_pos_local_x = object_pos_w[:, 0] - env_origins[:, 0]
    
    return (object_pos_local_x - 0.5).unsqueeze(-1)

def object_local_pos_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    object_pos_w = asset.data.root_pos_w
    env_origins = env.scene.env_origins
    # 相减得到局部坐标
    return object_pos_w - env_origins