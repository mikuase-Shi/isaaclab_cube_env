import torch
from isaaclab.envs import ManagerBasedRLEnv

def object_reached_goal(env: ManagerBasedRLEnv, pos_threshold: float, rot_threshold: float) -> torch.Tensor:
    import isaaclab.utils.math as math_utils

    object_pos_w = env.scene["object"].data.root_pos_w
    goal_pos_w = env.scene["goal"].data.root_pos_w
    
    object_quat_w = env.scene["object"].data.root_quat_w
    goal_quat_w = env.scene["goal"].data.root_quat_w
    
    # Distance
    pos_error = torch.norm(goal_pos_w - object_pos_w, p=2, dim=-1)
    
    # Rotation error
    object_quat_inv = math_utils.quat_inv(object_quat_w)
    quat_diff = math_utils.quat_mul(object_quat_inv, goal_quat_w)
    angle_diff = 2.0 * torch.acos(torch.clamp(quat_diff[:, 0], -1.0, 1.0))
    angle_diff = torch.where(angle_diff > torch.pi, 2 * torch.pi - angle_diff, angle_diff)
    
    is_close_pos = pos_error <= pos_threshold
    is_close_rot = angle_diff <= rot_threshold
    
    return is_close_pos & is_close_rot