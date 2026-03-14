import torch
from isaaclab.envs import ManagerBasedRLEnv


def ms_reaching_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    obj_pos = env.scene["object"].data.root_pos_w

    tcp_push_pose = obj_pos.clone()
    tcp_push_pose[:, 0] -= (0.05 + 0.005)  # -0.055 (Behind the cube)
    tcp_push_pose[:, 2] -= 0.05  # Lower part of the cube to prevent tipping

    tcp_to_push_pose_dist = torch.norm(tcp_push_pose - ee_pos, p=2, dim=-1)

    return 1.0 - torch.tanh(5.0 * tcp_to_push_pose_dist)


def ms_goal_reaching_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w

    tcp_push_pose = obj_pos.clone()
    tcp_push_pose[:, 0] -= 0.055
    tcp_push_pose[:, 2] -= 0.05
    tcp_to_push_pose_dist = torch.norm(tcp_push_pose - ee_pos, p=2, dim=-1)

    reach_multiplier = 1.0 - torch.tanh(10.0 * tcp_to_push_pose_dist) 
    
    # Distance to goal
    dist_to_goal = torch.norm(goal_pos - obj_pos, p=2, dim=-1)
    push_reward = 1.0 - torch.tanh(5.0 * dist_to_goal)

    return push_reward * reach_multiplier

def ms_goal_alignment_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    import isaaclab.utils.math as math_utils
    
    obj_quat = env.scene["object"].data.root_quat_w
    goal_quat = env.scene["goal"].data.root_quat_w
    obj_pos = env.scene["object"].data.root_pos_w
    goal_pos = env.scene["goal"].data.root_pos_w
    
    # Only reward alignment when close to the goal (e.g. within 10cm)
    dist_to_goal = torch.norm(goal_pos - obj_pos, p=2, dim=-1)
    close_to_goal_multiplier = 1.0 - torch.tanh(10.0 * dist_to_goal)
    
    # Calculate Z-axis rotation difference
    # q_diff = q_obj^{-1} * q_goal
    obj_quat_inv = math_utils.quat_inv(obj_quat)
    quat_diff = math_utils.quat_mul(obj_quat_inv, goal_quat)
    
    # Convert quaternion difference to axis-angle and extract angle magnitude
    # For small angles, angle roughly equals 2 * acos(w) or 2 * |xyz|
    angle_diff = 2.0 * torch.acos(torch.clamp(quat_diff[:, 0], -1.0, 1.0))
    # Normalize to [0, pi]
    angle_diff = torch.where(angle_diff > torch.pi, 2 * torch.pi - angle_diff, angle_diff)
    
    alignment_reward = 1.0 - torch.tanh(2.0 * angle_diff)
    
    return alignment_reward * close_to_goal_multiplier


def ms_z_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
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

