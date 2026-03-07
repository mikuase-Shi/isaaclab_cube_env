import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv

def object_flipped(env: ManagerBasedRLEnv, threshold: float = 0.5) -> torch.Tensor:
    """
    Terminates the episode when the object has successfully tipped over (flipped).
    This is determined by checking its local Z-axis relative to the world frame.
    If the absolute Z component is below the threshold, it is considered flipped (laying on its side).
    """
    quat = env.scene["object"].data.root_quat_w
    base_z = torch.zeros((quat.shape[0], 3), device=quat.device)
    base_z[:, 2] = 1.0
    
    world_z_of_object = math_utils.quat_apply(quat, base_z)
    
    # If the Z component of the object's local Z vector is near 0, it has tipped over
    return torch.abs(world_z_of_object[:, 2]) < threshold
