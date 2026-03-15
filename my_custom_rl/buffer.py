import torch

class RolloutBuffer:
    def __init__(self,num_envs,num_steps,obs_dim,act_dim,device="cuda:0"):
        self.num_envs=num_envs
        self.num_steps=num_steps
        self.device=device
        
        self.obs=torch.zeros((num_steps,num_envs,obs_dim),device=device)
        self.actions=torch.zeros((num_steps,num_envs,act_dim),device=device)
        self.rewards=torch.zeros((num_steps,num_envs),device=device)
        self.dones=torch.zeros((num_steps,num_envs),device=device)
        self.values=torch.zeros((num_steps,num_envs),device=device)
        self.log_probs=torch.zeros((num_steps,num_envs),device=device)
        
        self.advantages=torch.zeros((num_steps,num_envs),device=device)
        self.returns=torch.zeros((num_steps,num_envs),device=device)
        
        self.step=0

    def add(self,obs,action,reward,done,value,log_prob):
        self.obs[self.step]=obs
        self.actions[self.step]=action
        self.rewards[self.step]=reward
        self.dones[self.step]=done
        self.values[self.step]=value.squeeze(-1)
        self.log_probs[self.step]=log_prob
        self.step=(self.step+1)%self.num_steps
    
    def compute_gae(self,last_value,gamma=0.99,lam=0.95):
        advantage=0
        for t in reversed(range(self.num_steps)):
            if t==self.num_steps-1:
                next_non_terminal=1.0-self.dones[t]
                next_values=last_value.squeeze(-1)
            else:
                next_non_terminal=1.0-self.dones[t]
                next_values=self.values[t+1]
            
            delta=self.rewards[t]+gamma*next_values*next_non_terminal-self.values[t]
            advantage=delta+gamma*lam*next_non_terminal*advantage
            self.advantages[t]=advantage
        self.returns=self.advantages+self.values
        self.advantages=(self.advantages-self.advantages.mean())/(self.advantages.std()+1e-8)
    
    def get_generator(self, batch_size):
        flat_obs = self.obs.view(-1, self.obs.shape[-1])
        flat_actions = self.actions.view(-1, self.actions.shape[-1])
        flat_values = self.values.view(-1)
        flat_returns = self.returns.view(-1)
        flat_log_probs = self.log_probs.view(-1)
        flat_advantages = self.advantages.view(-1)
        
        total_samples = self.num_envs * self.num_steps
        indices = torch.randperm(total_samples, device=self.device)
        
        for start_idx in range(0, total_samples, batch_size):
            idx = indices[start_idx : start_idx + batch_size]
            yield (
                flat_obs[idx], flat_actions[idx], flat_values[idx],
                flat_returns[idx], flat_log_probs[idx], flat_advantages[idx]
            )