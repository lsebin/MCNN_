import numpy as np
import torch
import torch.nn as nn
import gym

from copy import deepcopy
from typing import Dict, Union, Tuple
from offlinerlkit.policy import BasePolicy


class MemIQLPolicy(BasePolicy):
    """
    Implicit Q-Learning <Ref: https://arxiv.org/abs/2110.06169>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic_q1: nn.Module,
        critic_q2: nn.Module,
        critic_v: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic_q1_optim: torch.optim.Optimizer,
        critic_q2_optim: torch.optim.Optimizer,
        critic_v_optim: torch.optim.Optimizer,
        unnorm_nodes_obs : np.ndarray,
        norm_nodes_obs : torch.Tensor,
        nodes_actions : torch.Tensor,
        device : str,
        scaler : object,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float  = 0.99,
        expectile: float = 0.8,
        temperature: float = 0.1,
        adv_normalized : bool = False,
        beta : float = 0.05,
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic_q1, self.critic_q1_old = critic_q1, deepcopy(critic_q1)
        self.critic_q1_old.eval()
        self.critic_q2, self.critic_q2_old = critic_q2, deepcopy(critic_q2)
        self.critic_q2_old.eval()
        self.critic_v = critic_v

        self.actor_optim = actor_optim
        self.critic_q1_optim = critic_q1_optim
        self.critic_q2_optim = critic_q2_optim
        self.critic_v_optim = critic_v_optim

        self.action_space = action_space
        self._tau = tau
        self._gamma = gamma
        self._expectile = expectile
        self._temperature = temperature
        self.adv_normalized = adv_normalized
        self.beta = beta
        self.unnorm_nodes_obs = unnorm_nodes_obs
        self.norm_nodes_obs = norm_nodes_obs
        self.nodes_actions = nodes_actions
        self.device = device
        self.scaler = scaler

    def train(self) -> None:
        self.actor.train()
        self.critic_q1.train()
        self.critic_q2.train()
        self.critic_v.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic_q1.eval()
        self.critic_q2.eval()
        self.critic_v.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_q1_old.parameters(), self.critic_q1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic_q2_old.parameters(), self.critic_q2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        _, closest_nodes = torch.cdist(torch.from_numpy(obs).float().to(self.device), self.unnorm_nodes_obs).min(dim=1)
        mem_actions = self.nodes_actions[closest_nodes, :]
        obs = self.scaler.transform(obs)
        obs = torch.from_numpy(obs).float().to(self.device)
        distance = torch.norm(obs - self.norm_nodes_obs[closest_nodes, :], p=2, dim=1).unsqueeze(1)
        with torch.no_grad():
            dist = self.actor(obs, mem_actions, distance, beta=0)
            action = dist.cpu().numpy()
        action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        return action
    
    def _expectile_regression(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self._expectile, (1 - self._expectile))
        return weight * (diff**2)
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, discounted_return, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch['sum_rewards'], batch["terminals"]
        mem_obss, mem_next_obss, mem_actions, mem_rewards, mem_sum_rewards = batch['mem_observations'], batch['mem_next_observations'], batch['mem_actions'], batch['mem_rewards'], batch['mem_sum_rewards']
        distance = torch.norm(obss - mem_obss, p=2, dim=1).unsqueeze(1)
        next_distance = torch.norm(next_obss - mem_next_obss, p=2, dim=1).unsqueeze(1)
        
        # update value net
        with torch.no_grad():
            q1, q2 = self.critic_q1_old(obss, actions), self.critic_q2_old(obss, actions)
            q = torch.min(q1, q2)
        v = self.critic_v(obss, mem_sum_rewards, distance, beta=0)
        critic_v_loss = self._expectile_regression(q-v).mean()
        self.critic_v_optim.zero_grad()
        critic_v_loss.backward()
        self.critic_v_optim.step()

        # update critic
        q1, q2 = self.critic_q1(obss, actions), self.critic_q2(obss, actions)
        with torch.no_grad():
            next_mem_sum_rewards = mem_sum_rewards - mem_rewards
            next_v = self.critic_v(next_obss, next_mem_sum_rewards, next_distance, beta=0)
            target_q = rewards + self._gamma * (1 - terminals) * next_v
        
        critic_q1_loss = ((q1 - target_q).pow(2)).mean()
        critic_q2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic_q1_optim.zero_grad()
        critic_q1_loss.backward()
        self.critic_q1_optim.step()

        self.critic_q2_optim.zero_grad()
        critic_q2_loss.backward()
        self.critic_q2_optim.step()

        # update actor
        with torch.no_grad():
            q1, q2 = self.critic_q1_old(obss, actions), self.critic_q2_old(obss, actions)
            q = torch.min(q1, q2)
            v = self.critic_v(obss, mem_sum_rewards, distance, beta=0)
            exp_a = torch.exp((q - v) * self._temperature)
            exp_a = torch.clip(exp_a, None, 100.0)
        cur_policy = self.actor(obss, mem_actions, distance, beta=0)
        actor_loss = (exp_a * (cur_policy - actions).pow(2)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._sync_weight()

        return {
            "loss/actor": actor_loss.item(),
            "loss/q1": critic_q1_loss.item(),
            "loss/q2": critic_q2_loss.item(),
            "loss/v": critic_v_loss.item()
        }