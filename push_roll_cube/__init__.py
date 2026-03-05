import gymnasium as gym
from . import amazon_chess_env_cfg

gym.register(
    id="Isaac-Push-Flip-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amazon_chess_env_cfg:PushEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PushPPORunnerCfg",
    },
)