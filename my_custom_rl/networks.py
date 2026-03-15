import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorNet(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_dims=[256,128,64]):
        super().init()
        layers=[]
        last_dim=obs_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim,dim))
            layers.append(nn.Tanh())
            last_di=dim
        
        self.net=nn.Sequential(*layers)
        self.action_mean=nn.Linear(last_dim,act_dim)

        self.action_log_std=nn.Parameter(torch.zeros(1,act_dim))

    def forward(self,obs):
        x=self.net(obs)
        mean=self.action_mean(x)
        std=self.action_log_std.exp().expand_as(mean)
        
        return Normal(mean,std)

class CriticNet(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_dims=[256,128,64]):
        super().init()
        layers=[]
        last_dim=obs_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim,dim))
            layers.append(nn.Tanh())
            last_di=dim
        
        self.net=nn.Sequential(*layers)
        self.value_head=nn.Linear(last_dim,1)
    
    def forward(self,obs):
        x=self.net(obs)
        return self.value_head(x)
        
        