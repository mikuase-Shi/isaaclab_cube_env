import torch
import torch.nn as nn
from .networks import ActorNet,CriticNet
from .buffer import RolloutBuffer

class PPOAgent:
    def __init__(self,num_envs,obs_dim,act_dim,num_steps=24,device="cuda:0"):
        self.device=device
        self.actor=ActorNet(obs_dim,act_dim).to(device)
        self.critic=CriticNet(obs_dim).to(device)
        self.optimizer=torch.optim.Adam([
            {'params':self.actor.parameters(),'lr':1e-3},
            {'params':self.critic.parameters(),'lr':1e-3}
        ])
        self.buffer=RolloutBuffer(num_envs,num_steps,obs_dim,act_dim,device)
        self.clip_param=0.2
        self.entropy_coef=0.005
    
    @torch.no_grad()
    def select_action(self,obs):
        action=self.action(obs).sample()
        log_prob=self.action(obs).log_prob(action).sum(dim=-1)
        value=self.critic(obs)
        return action,log_prob,value
    
    def update(self,num_epochs=5,batch_size=4096):
        for _ in range(num_epochs):
            for obs,actions,old_values,returns,old_log_probs,advantages in self.buffer.get_generator(batch_size):
                dist=self.actor(obs)
                new_log_probs=dist.log_prob(actions).sum(dim=-1)
                entropy=dist.entropy().sum(dim=-1).mean()
                new_values=self.critic(obs).squeeze(-1)
                ratio=torch.exp(new_log_probs-old_log_probs)
                surr1=ratio*advantages
                surr2=(torch.clamp(ratio,1-self.clip_param,1+self.clip_param)*advantages)
                actor_loss=-torch.min(surr1,surr2).mean()
                critic_loss=nn.MSELoss()(new_values,returns)
                loss=actor_loss+0.5*critic_loss-self.entropy_coef*entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),1.0)

                self.optimizer.step()
                