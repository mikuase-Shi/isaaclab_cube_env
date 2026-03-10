"""ManiSkill-style dense reward shaping for Push-Forward Cube task.

Adopts the philosophy of ManiSkill's PushCube:
- Pre-push pose alignment via virtual anchor point
- Strict Z-axis grounding (anti-lifting/anti-tipping)
- Forward pushing velocity reward
- Lateral drift penalty (straight-line "rails")
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Pre-push Pose Alignment (Reaching Stage)
# -----------------------------------------------------------------------------


def pre_push_distance_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize distance between EE and a virtual anchor point behind the cube.

    The anchor is 0.06m behind the cube along world -X, at the bottom height
    of the cube. Guides the arm to the correct pre-push position.

    Returns:
        L2 norm between EE position and virtual anchor.
    """
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    cube_pos = env.scene["object"].data.root_pos_w

    # Anchor: 0.06m behind cube (world -X), at bottom of cube (center z=0.15, half-height=0.1 -> bottom at 0.05)
    target_pos = cube_pos + torch.tensor([-0.06, 0.0, -0.1], device=cube_pos.device)

    distance = torch.norm(ee_pos - target_pos, p=2, dim=-1)
    return distance


# -----------------------------------------------------------------------------
# Z-Axis Grounding (Anti-Lifting / Anti-Tipping)
# -----------------------------------------------------------------------------


def cube_z_lift_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Strictly penalize cube Z above its nominal table height.

    Prevents RL agents from cheating by lifting or flipping the cube.
    Initial cube center Z is 0.15.

    Returns:
        max(0, current_cube_z - 0.15) — penalizes any upward movement.
    """
    cube_z = env.scene["object"].data.root_pos_w[:, 2]
    return torch.clamp(cube_z - 0.15, min=0.0)


# -----------------------------------------------------------------------------
# Forward Pushing Reward (Moving Stage)
# -----------------------------------------------------------------------------


def push_forward_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward cube linear velocity along world +X axis.

    Primary driving force for the push task.

    Returns:
        object_lin_vel[:, 0] (X-component of cube velocity).
    """
    object_lin_vel = env.scene["object"].data.root_lin_vel_w
    return object_lin_vel[:, 0]


# -----------------------------------------------------------------------------
# Lateral Drift Penalty (Straight-Line "Rails")
# -----------------------------------------------------------------------------


def lateral_drift_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize Y-axis position deviation and Y-axis velocity.

    Forces straight-line pushing along X. ManiSkill "rails" logic.

    Returns:
        y_drift = square(cube_pos_y) + 0.1 * square(cube_vel_y)
    """
    object_pos = env.scene["object"].data.root_pos_w
    object_lin_vel = env.scene["object"].data.root_lin_vel_w

    y_drift = torch.square(object_pos[:, 1]) + 0.1 * torch.square(object_lin_vel[:, 1])
    return y_drift


# -----------------------------------------------------------------------------
# Legacy / Auxiliary (kept for compatibility if used elsewhere)
# -----------------------------------------------------------------------------


def object_x_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Alias for push_forward_velocity. Reward object velocity along +X."""
    return push_forward_velocity(env)


def object_x_displacement(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Object X position minus 0.5 (initial X)."""
    object_pos = env.scene["object"].data.root_pos_w
    return object_pos[:, 0] - 0.5
