from torch import device
import torch
import torch.nn as nn
from torch.distributions import Normal
from dataclasses import dataclass

@dataclass
class PPOConfig:
    num_envs:int
    obs_dim: int
    act_dim: int
    num_steps: int = 24
    hidden_dims: tuple = (256, 128, 64)
    lr: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    num_epochs: int = 5
    batch_size: int = 4096
    max_grad_norm: float = 1.0
    device: str = "cuda:0"

def layer_init(layer, std=torch.sqrt(torch.tensor(2.0)), bias_const=0.0):
    """正交初始化：让深层网络在强化学习中更稳定地传递梯度"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self,cfg:PPOConfig):
        super().__init__()

        actor_layers=[]
        last_dim=cfg.obs_dim
        for h in cfg.hidden_dims:
            actor_layers.append(nn.Linear(last_dim,h))
            actor_layers.append(nn.ELU())
            last_dim=h
        self.actor_net=nn.Sequential(*actor_layers)
        self.actor_mean=layer_init(nn.Linear(last_dim,cfg.act_dim),std=0.01)
        self.actor_logstd=nn.Parameter(torch.zeros(1,cfg.act_dim))

        critic_layers = []
        last_dim = cfg.obs_dim
        for h in cfg.hidden_dims:
            critic_layers += [layer_init(nn.Linear(last_dim, h)), nn.ELU()]
            last_dim = h
        self.critic_net = nn.Sequential(*critic_layers)
        self.critic_value = layer_init(nn.Linear(last_dim, 1), std=1.0)
        
    def get_action_and_value(self,obs,action=None):
        hidden_actor=self.actor_net(obs)
        mean=self.actor_mean(hidden_actor)
        std=self.actor_logstd.exp().expand_as(mean)
        dist=Normal(mean,std)
        
        # 如果没有传入 action，说明是推断阶段   传入是训练 用action计算log_prob和entropy
        if action is None:
            action=dist.sample()
        
        value=self.critic_value(self.critic_net(obs)).squeeze(-1)
        return action,dist.log_prob(action).sum(-1),dist.entropy().sum(-1),value
        

class PPOAgent:
    def __init__(self,cfg:PPOConfig):
        self.cfg=cfg
        self.device=device
        self.net=ActorCritic(cfg).to(self.device)
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=cfg.lr,eps=1e-5)
        
        steps,envs=cfg.num_steps,cfg.num_envs
        self.buffers={
            k:torch.zeros(s,device=self.device) for k,s in[
                ('obs', (steps, envs, cfg.obs_dim)), 
                ('actions', (steps, envs, cfg.act_dim)),
                ('rewards', (steps, envs)), ('dones', (steps, envs)),
                ('values', (steps, envs)), ('logprobs', (steps, envs))
            ]
        }
        self.step=0
    
    @torch.no_grad()
    def select_action(self,obs):
        action,logprob,_,value=self.net.get_action_and_value(obs)
        return action,logprob,value
    
    def store_reansition(self,obs,action,reward,done,value,logprob):
        b=self.buffers
        b['obs'][self.step], b['actions'][self.step] = obs, action
        b['rewards'][self.step], b['dones'][self.step] = reward, done
        b['values'][self.step], b['logprobs'][self.step] = value, logprob
        self.step = (self.step + 1) % self.cfg.num_steps

    def update(self, next_obs, next_done):
        cfg, b = self.cfg, self.buffers
        
        with torch.no_grad():
            _, _, _, next_values = self.net.get_action_and_value(next_obs)
            advantages = torch.zeros_like(b['rewards'])
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_values
                else:
                    nextnonterminal = 1.0 - b['dones'][t + 1]
                    nextvalues = b['values'][t + 1]
                delta = b['rewards'][t] + cfg.gamma * nextvalues * nextnonterminal - b['values'][t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.lam * nextnonterminal * lastgaelam
            returns = advantages + b['values']

        b_obs = b['obs'].view(-1, cfg.obs_dim)
        b_actions = b['actions'].view(-1, cfg.act_dim)
        b_logprobs = b['logprobs'].view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        inds = torch.arange(cfg.batch_size * (cfg.num_envs * cfg.num_steps // cfg.batch_size), device=self.device)
        
        for epoch in range(cfg.num_epochs):
            torch.randperm(len(inds), out=inds)
            for start in range(0, len(inds), cfg.batch_size):
                mb_inds = inds[start:start + cfg.batch_size]
                
                _, newlogprob, entropy, newvalue = self.net.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_inds]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_param, 1 + cfg.clip_param)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                ent_loss = entropy.mean()

                loss = pg_loss - cfg.entropy_coef * ent_loss + cfg.value_coef * v_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()