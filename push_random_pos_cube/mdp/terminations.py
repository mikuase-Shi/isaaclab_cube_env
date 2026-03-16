import torch
from isaaclab.envs import ManagerBasedRLEnv

def object_reached_goal(env: ManagerBasedRLEnv, pos_threshold: float, rot_threshold: float) -> torch.Tensor:
    """Success if object XY is close enough to goal, ignore orientation.

    rot_threshold is kept in the signature for backward compatibility but unused.
    """
    object_pos_w = env.scene["object"].data.root_pos_w
    goal_pos_w = env.scene["goal"].data.root_pos_w

    pos_error_2d = torch.norm(goal_pos_w[:, :2] - object_pos_w[:, :2], p=2, dim=-1)
    is_close_pos = pos_error_2d <= pos_threshold
    return is_close_pos