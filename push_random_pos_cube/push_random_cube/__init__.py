import gymnasium as gym

gym.register(
    id="Isaac-Push-Random-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_random_cube_env:PushEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PushPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Push-Random-Franka-SAC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_random_cube_env:PushEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_sac_cfg:PushSACRunnerCfg",
    },
)

gym.register(
    id="Isaac-Push-Random-Franka-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_random_cube_env:PushEnvCfg",
        "my_custom_rl_cfg_entry_point": f"{__name__}.agents.my_custom_agent_cfg:MyCustomAgentCfg",
    },
)
