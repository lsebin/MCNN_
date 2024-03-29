import numpy as np
import torch
import random, d4rl, gym

from typing import Optional, Union, Tuple, Dict


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu",
        is_awr: bool = False,
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)
        self.mem_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.mem_next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.mem_actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.mem_rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        
        self.is_awr = is_awr
        if is_awr:
            self.mem_sum_rewards = np.zeros((self._max_size, 1), dtype=np.float32)
            self.sum_rewards = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()
        self.mem_observations[self._ptr] = np.array(obs).copy()
        self.mem_next_observations[self._ptr] = np.array(next_obs).copy()
        self.mem_actions[self._ptr] = np.array(action).copy()
        self.mem_rewards[self._ptr] = np.array(reward).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()
        self.mem_observations[indexes] = np.array(obss).copy()
        self.mem_next_observations[indexes] = np.array(next_obss).copy()
        self.mem_actions[indexes] = np.array(actions).copy()
        self.mem_rewards[indexes] = np.array(rewards).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
        
 
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)
        mem_observations = np.array(dataset["mem_observations"], dtype=self.obs_dtype)
        mem_next_observations = np.array(dataset["mem_next_observations"], dtype=self.obs_dtype)
        mem_actions = np.array(dataset["mem_actions"], dtype=self.action_dtype)
        mem_rewards = np.array(dataset["mem_rewards"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.mem_observations = mem_observations
        self.mem_next_observations = mem_next_observations
        self.mem_actions = mem_actions
        self.mem_rewards = mem_rewards

        self._ptr = len(observations)
        self._size = len(observations)
        
        if self.is_awr:
            mem_sum_rewards = np.array(dataset["mem_sum_rewards"], dtype=np.float32).reshape(-1, 1)
            sum_rewards = np.array(dataset["sum_rewards"], dtype=np.float32).reshape(-1, 1)
            self.mem_sum_rewards = mem_sum_rewards
            self.sum_rewards = sum_rewards

    # For ensemble dynamics -> use the load_datset above since it does not need mem_
    
    def load_dataset_original(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations) 
    
    
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        self.mem_observations = (self.mem_observations - mean) / std
        self.mem_next_observations = (self.mem_next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        sample_dataset = {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
            "mem_observations": torch.tensor(self.mem_observations[batch_indexes]).to(self.device),
            "mem_actions": torch.tensor(self.mem_actions[batch_indexes]).to(self.device),
            "mem_next_observations": torch.tensor(self.mem_next_observations[batch_indexes]).to(self.device),
            "mem_rewards": torch.tensor(self.mem_rewards[batch_indexes]).to(self.device),
        }
        
        if self.is_awr:
            sample_dataset.update({
                "mem_sum_rewards": torch.tensor(self.mem_sum_rewards[batch_indexes]).to(self.device),
                "sum_rewards" : torch.tensor(self.sum_rewards[batch_indexes]).to(self.device),
            })
            
        return sample_dataset

    def sample_paths(self, num_paths: int) -> Dict[str, torch.Tensor]:
        indices_where_paths_end = np.where(self.terminals == 1)[0]
        indices_where_paths_start = np.concatenate([[0], indices_where_paths_end[:-1] + 1])
        indices_per_path = [list(range(s, e+1, 1)) for s, e in zip(indices_where_paths_start, indices_where_paths_end)]
        sampled_paths = random.sample(indices_per_path, num_paths)
        sampled_idxs = np.concatenate(sampled_paths)

        sample_dataset = {
            "observations": torch.tensor(self.observations[sampled_idxs]).to(self.device),
            "actions": torch.tensor(self.actions[sampled_idxs]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[sampled_idxs]).to(self.device),
            "terminals": torch.tensor(self.terminals[sampled_idxs]).to(self.device),
            "rewards": torch.tensor(self.rewards[sampled_idxs]).to(self.device),
            "mem_observations": torch.tensor(self.mem_observations[sampled_idxs]).to(self.device),
            "mem_actions": torch.tensor(self.mem_actions[sampled_idxs]).to(self.device),
            "mem_next_observations": torch.tensor(self.mem_next_observations[sampled_idxs]).to(self.device),
            "mem_rewards": torch.tensor(self.mem_rewards[sampled_idxs]).to(self.device),
        }

        if self.is_awr:
            sample_dataset.update({
                "mem_sum_rewards": torch.tensor(self.mem_sum_rewards[sampled_idxs]).to(self.device),
            })  

        return sample_dataset

    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy(),
            "mem_observations": self.mem_observations[:self._size].copy(),
            "mem_actions": self.mem_actions[:self._size].copy(),
            "mem_next_observations": self.mem_next_observations[:self._size].copy(),
            "mem_rewards": self.mem_rewards[:self._size].copy(),
        }