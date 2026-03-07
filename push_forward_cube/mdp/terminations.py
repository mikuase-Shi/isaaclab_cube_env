import torch
from isaaclab.envs import ManagerBasedRLEnv

def object_moved_x_cm(env: ManagerBasedRLEnv, distance: float) -> torch.Tensor:
    """
    Terminates the episode when the object has moved forward along the X-axis by a specific distance.
    The movement is relative to the object's initial X position (which is 0.5 as defined in the scene).
    """
    object_pos_x = env.scene["object"].data.root_pos_w[:, 0]
    # Check if displaced by given distance (initial position is 0.5)
    return object_pos_x - 0.5 >= distance
