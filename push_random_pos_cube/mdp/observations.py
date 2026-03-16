import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

def rel_ee_object_distance(env:ManagerBasedRLEnv)->torch.Tensor:
    ee_pos=env.scene["ee_frame"].data.target_pos_w[...,0,:]
    object_pos=env.scene["object"].data.root_pos_w
    return object_pos-ee_pos

def object_to_goal_pos_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    object_pos_w = env.scene["object"].data.root_pos_w
    goal_pos_w = env.scene["goal"].data.root_pos_w
    return goal_pos_w - object_pos_w

def object_to_goal_quat_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    object_quat_w = env.scene["object"].data.root_quat_w
    goal_quat_w = env.scene["goal"].data.root_quat_w
    # q_diff = q_obj^{-1} * q_goal
    object_quat_inv = math_utils.quat_inv(object_quat_w)
    quat_diff = math_utils.quat_mul(object_quat_inv, goal_quat_w)
    return quat_diff

def object_local_pos_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    object_pos_w = asset.data.root_pos_w
    env_origins = env.scene.env_origins
    
    return object_pos_w - env_origins