import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import SACPolicy
from offlinerlkit.dynamics import BaseDynamics


class MNMPolicy(SACPolicy):
    """
    Model-based Offline Policy Optimization <Ref: https://arxiv.org/abs/2005.13239>
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.dynamics = dynamics
        
    """ 
    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        raw_rewards_arr = np.array([])
        penalty_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            actions = self.select_action(observations)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())
            raw_rewards_arr = np.append(raw_rewards_arr, info["raw_reward"].flatten())
            penalty_arr = np.append(penalty_arr, info["penalty"].flatten()) # OR MODIFY PENALTY HERE!

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean(), "raw_reward_mean": raw_rewards_arr.mean(), "penalty_mean": penalty_arr.mean()}
    """
    
    def rollout(
        self,
        init_obss: np.ndarray,
        real_next_obs: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        raw_rewards_arr = np.array([])
        penalty_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            # this is coming from policy -> have to replace with real_action(from real_buffer)
            #actions = real_actions # change to real action
            actions = self.select_action(observations)
            # need to also input real rewards to the step function
            next_observations, rewards, terminals, info = self.dynamics.step(observations, real_next_obs, actions)
            
            
            # replace with real_values
            # real transition with penalty from rollout
            rollout_transitions["obss"].append(observations)
            # next_obs/actions/terminals into real
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())
            raw_rewards_arr = np.append(raw_rewards_arr, info["raw_reward"].flatten())
            penalty_arr = np.append(penalty_arr, info["penalty"].flatten()) # OR MODIFY PENALTY HERE!

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean(), "raw_reward_mean": raw_rewards_arr.mean(), "penalty_mean": penalty_arr.mean()}
         
         
             
    def add_penalties(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Dict[str, np.ndarray]:
        
        rollout_transitions = defaultdict(list)

        # rollout
        #observations = init_obss["observations"]
        #actions = init_obss["actions"]
        for observations, actions, next_observations, _, terminals in init_obss:
            # have to return updated rewards
            next_observations, rewards, info = self.dynamics.step(observations, next_observations, actions)
            
            # replace with real_values
            # real transition with penalty from rollout
            rollout_transitions["obss"].append(observations)
            # next_obs/actions/terminals into real
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions
           # {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean(), "raw_reward_mean": raw_rewards_arr.mean(), "penalty_mean": penalty_arr.mean()}

    def learn(self, batch: Dict) -> Dict[str, float]:
        #real_batch, fake_batch = batch["real"], batch["fake"]
        #mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}
        mix_batch = batch["real"]
        return super().learn(mix_batch)
