import torch
import gymnasium as gym

# 1. 导入你的物理环境，完成 IsaacLab 引擎注册
import push_forward_cube 
# 2. 导入咱们刚手搓的 PPO 大脑
from my_custom_rl.agent import PPOAgent 

def main():
    device = "cuda:0"
    num_envs = 4096
    num_steps_per_env = 24
    max_iterations = 10000

    print("🌍 正在召唤 IsaacLab 物理世界...")
    # 彻底告别 rsl_rl，纯净启动你的环境！
    env = gym.make("Isaac-Push-Forward-Franka-v0", num_envs=num_envs, device=device)
    
    # 获取观测维度和动作维度
    obs_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]
    
    print("🧠 正在加载手搓版 PPO 大脑...")
    agent = PPOAgent(num_envs, obs_dim, act_dim, num_steps=num_steps_per_env, device=device)

    # 获取初始画面 (记得剥开字典！)
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    print("🚀 开始狂飙模式炼丹！")
    for iteration in range(max_iterations):
        # 收集数据阶段 (Rollout)
        for step in range(num_steps_per_env):
            # 大脑思考
            actions, log_probs, values = agent.select_action(obs)
            
            # 物理世界步进
            next_obs_dict, rewards, dones, truncated, info = env.step(actions)
            next_obs = next_obs_dict["policy"]
            
            # 存入记忆库
            agent.buffer.add(obs, actions, rewards, dones, values, log_probs)
            
            obs = next_obs

        # 一轮数据收集完毕，准备进化！
        # 1. 预测最后一步的价值，用于计算 GAE
        with torch.no_grad():
            last_value = agent.critic(obs)
        agent.buffer.compute_gae(last_value)
        
        # 2. 大脑开始通过反向传播自我学习
        agent.update()
        
        # 打印日志（这里只打印个平均奖励看看效果）
        mean_reward = agent.buffer.rewards.mean().item()
        if iteration % 10 == 0:
            print(f"[{iteration}/{max_iterations}] Mean Reward: {mean_reward:.4f}")

    print("🎉 训练完成！")
    env.close()

if __name__ == "__main__":
    main()