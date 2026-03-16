class MyCustomAgentCfg:
    """Configuration class for the custom RL agent."""
    num_envs = 4096
    num_steps = 24
    device = "cuda:0"
    
    # Network hyperparameters
    hidden_dims = (256, 128, 64)
    lr = 1e-3
    gamma = 0.99
    lam = 0.95
    
    # PPO specific hyperparameters
    clip_param = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    num_epochs = 5
    batch_size = 4096
    max_grad_norm = 1.0

    @classmethod
    def get_ppo_config(cls, obs_dim: int, act_dim: int):
        try:
            from my_custom_rl.ppo_agent import PPOConfig
            return PPOConfig(
                num_envs=cls.num_envs,
                obs_dim=obs_dim,
                act_dim=act_dim,
                num_steps=cls.num_steps,
                hidden_dims=cls.hidden_dims,
                lr=cls.lr,
                gamma=cls.gamma,
                lam=cls.lam,
                clip_param=cls.clip_param,
                entropy_coef=cls.entropy_coef,
                value_coef=cls.value_coef,
                num_epochs=cls.num_epochs,
                batch_size=cls.batch_size,
                max_grad_norm=cls.max_grad_norm,
                device=cls.device
            )
        except ImportError:
            # Fallback if PPOConfig is not available, just return this class
            return cls
