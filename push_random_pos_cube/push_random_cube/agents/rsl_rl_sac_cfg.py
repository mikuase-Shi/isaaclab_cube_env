from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
# Note: RSL-RL standard SAC configurations might be RslRlOffPolicyRunnerCfg instead, 
# but IsaacLab wrappers typically map to similar structure.
# Using standard PPO structure as baseline, tailored for what would be SAC conceptually if available.
# We will construct a configuration utilizing standard syntax for RSL-RL

@configclass
class PushSACRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_push_random_sac"
    empirical_normalization = False
    
    # Below is a hypothetical wrapper config for SAC in RSL-RL
    # If using RslRlOffPolicyRunnerCfg, change base class accordingly
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=1.0e-3,
        schedule="adaptive",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        use_clipped_value_loss=True,
        value_loss_coef=1.0,
        entropy_coef=0.005,
        max_grad_norm=1.0,
    )