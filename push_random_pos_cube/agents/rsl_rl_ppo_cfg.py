from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class PushPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24 
    max_iterations = 1500 
    save_interval = 50 
    experiment_name = "franka_push_random_pos" 
    # 打开经验归一化，稳定观测与回报尺度
    empirical_normalization = True 

    policy = RslRlPpoActorCriticCfg(
        # 降低初始动作噪声，避免一开始策略太激进
        init_noise_std=0.3,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu", 
    )

    algorithm = RslRlPpoAlgorithmCfg(
        # 略降学习率，减少更新过大导致的数值不稳定
        learning_rate=3.0e-4,
        schedule="adaptive",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        use_clipped_value_loss=True,
        value_loss_coef=1.0,
        # 略减小熵系数，降低过度探索导致的策略发散
        entropy_coef=0.001,
        max_grad_norm=1.0,
    )