import torch
from isaaclab.envs import ManagerBasedRLEnv

def rel_ee_object_distance(env:ManagerBasedRLEnv)->torch.Tensor:
    ee_pos=env.scene["ee_frame"].data.target_pos_w[...,0,:]
    object_pos=env.scene["object"].data.root_pos_w

    return object_pos-ee_pos