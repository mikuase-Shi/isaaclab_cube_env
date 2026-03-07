# IsaacLab Custom Environments: Push Cube

This repository contains custom Reinforcement Learning environments built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab) for training a Franka Panda robot to interact with a cube.

## Environments

There are current two main environments in this repository:

1. **`push_roll_cube`**
   - **Task**: `Isaac-Push-Flip-Franka-v0`
   - **Description**: The robot must push and roll the cube.

2. **`push_forward_cube`**
   - **Task**: `Isaac-Push-Forward-Franka-v0`
   - **Description**: The robot must push the cube forward without causing it to fall over. This environment includes a specific reward penalty (`ee_object_z_distance_penalty`) that encourages the robot's end-effector to stay aligned with the central z-height of the cube to maintain stability.

## How to Run

Before running the environments, ensure you have IsaacLab installed and the `isaaclab_root` path in the launcher scripts correctly points to your IsaacLab installation (default is `/root/IsaacLab`).

### Running `push_roll_cube`
To train the `push_roll_cube` environment, use the provided custom launcher:

```bash
python custom_launcher.py
```

### Running `push_forward_cube`
To train the `push_forward_cube` environment, use the provided custom launcher:

```bash
python custom_launcher_forward.py
```

Both launchers will automatically navigate to your IsaacLab directory, register the corresponding custom environment ID, and invoke `rsl_rl/train.py` with 4096 environments in headless mode.

## Structure & Imports

Each environment is structured as a module that is imported before the RL runner starts. The launchers handle the importing automatically:

```python
# For the push_roll_cube environment
import push_roll_cube

# For the push_forward_cube environment
import push_forward_cube
```

Importing these modules automatically registers the custom gym environment IDs (`Isaac-Push-Flip-Franka-v0` and `Isaac-Push-Forward-Franka-v0`) with IsaacLab's RL environment manager.

### Key Configuration Files
- `[env_name]/push_cube_env_cfg.py`: Contains the main environment configuration, defining the robot, objects, rewards, observations, and terminations.
- `[env_name]/mdp/rewards.py`: Contains the custom reward functions (like `ee_object_z_distance_penalty` or `object_local_z_alignment`).
- `[env_name]/mdp/observations.py`: Defines the observation functions.
- `[env_name]/agents/rsl_rl_ppo_cfg.py`: Contains the PPO hyperparameters used for training by RSL-RL.
