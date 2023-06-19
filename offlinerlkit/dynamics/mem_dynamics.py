import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from torch.utils.data.dataloader import DataLoader
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from offlinerlkit.modules import MemDynamicsModel
from copy import deepcopy
import time
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset

class MemDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray, mem_inputs: np.ndarray, mem_targets: np.ndarray) -> None:
        super().__init__()
        self.inputs = inputs
        self.targets = targets
        self.mem_inputs = mem_inputs
        self.mem_targets = mem_targets

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.inputs[idx], self.targets[idx], self.mem_inputs[idx], self.mem_targets[idx]

class MemDynamics(object):
    def __init__(
        self,
        model: MemDynamicsModel,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        dataset: Dict,
        penalty_coef: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optim
        self.scheduler = None
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self.device = self.model.device
        self.penalty_coef = penalty_coef
        self.len_of_beta_step = 15
        self.num_beta_steps = 15
        self.num_epochs_bw_evals = 1

        self.nodes_obs = torch.from_numpy(dataset["memories_obs"]).float().to(self.device)
        self.nodes_actions = torch.from_numpy(dataset["memories_actions"]).float().to(self.device)
        self.nodes_next_obs = torch.from_numpy(dataset["memories_next_obs"]).float().to(self.device)
        self.nodes_rewards = torch.from_numpy(dataset["memories_rewards"]).float().to(self.device).unsqueeze(1)
        self.nodes_inputs = torch.cat([self.nodes_obs, self.nodes_actions], dim=1)
        self.nodes_next_inputs = torch.cat([self.nodes_obs], dim=1)
        
        # new for normalization
        self.obss_mean = np.mean(dataset["observations"], axis=0, keepdims=True)
        self.obss_std = np.std(dataset["observations"], axis=0, keepdims=True)
        self.obss_mean_tensor = torch.from_numpy(self.obss_mean).float().to(self.device)
        self.obss_std_tensor = torch.from_numpy(self.obss_std).float().to(self.device)
        
        self.actions_mean = np.mean(dataset["actions"], axis=0, keepdims=True)
        self.actions_std = np.std(dataset["actions"], axis=0, keepdims=True)
        self.actions_mean_tensor = torch.from_numpy(self.actions_mean).float().to(self.device)
        self.actions_std_tensor = torch.from_numpy(self.actions_std).float().to(self.device)
        
        self.abs_max_delta_obss = np.max(np.abs(dataset["next_observations"]-dataset["observations"]), axis=0, keepdims=True)
        self.abs_max_delta_obss_tensor = torch.from_numpy(self.abs_max_delta_obss).float().to(self.device)
        
        self.rewards_mean = np.mean(dataset["rewards"], axis=0, keepdims=True)
        self.rewards_mean_tensor = torch.from_numpy(self.rewards_mean).float().to(self.device)
        self.abs_max_rewards = np.max(np.abs(dataset["rewards"]), axis=0, keepdims=True)
        self.abs_max_rewards_tensor = torch.from_numpy(self.abs_max_rewards).float().to(self.device)
        self.abs_max_rewards_diff = np.max(np.abs(dataset["rewards"]-self.rewards_mean), axis=0, keepdims=True)
        self.abs_max_rewards_diff_tensor = torch.from_numpy(self.abs_max_rewards_diff).float().to(self.device)
        
        self.rewards_std = np.std(dataset["rewards"], axis=0, keepdims=True)
        self.rewards_std_tensor = torch.from_numpy(self.rewards_std).float().to(self.device)
        self.abs_max_rewards = np.max(np.abs(dataset["rewards"]), axis=0, keepdims=True)
        self.abs_max_rewards_tensor = torch.from_numpy(self.abs_max_rewards).float().to(self.device)
        
        self.diagnostics = {}

    def find_memories(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, closest_nodes = torch.cdist(inputs, self.nodes_inputs).min(dim=1)
        mem_obss = self.nodes_obs[closest_nodes, :]
        mem_actions = self.nodes_actions[closest_nodes, :]
        mem_next_obss = self.nodes_next_obs[closest_nodes, :]
        mem_rewards = self.nodes_rewards[closest_nodes, :]

        delta_mem_obss = (mem_next_obss - mem_obss) / self.abs_max_delta_obss_tensor

        mem_obss = (mem_obss - self.obss_mean_tensor) / self.obss_std_tensor
        mem_next_obss = (mem_next_obss - self.obss_mean_tensor) / self.obss_std_tensor
        # mem_actions = (mem_actions - self.actions_mean_tensor) / self.actions_std_tensor
        mem_rewards = (mem_rewards - self.rewards_mean_tensor)/ self.abs_max_rewards_diff_tensor
        
        mem_inputs = torch.cat([mem_obss, mem_actions], dim=1)
        mem_targets = torch.cat([delta_mem_obss, mem_rewards], dim=1)
        return mem_inputs, mem_targets
    
    def find_memories_penalty (self, inputs: torch.Tensor) -> np.ndarray:
        _, closest_nodes = torch.cdist(inputs, self.nodes_next_inputs).min(dim=1)	   
        mem_obss = self.nodes_obs[closest_nodes, :]	 
        mem_obss = (mem_obss - self.obss_mean_tensor) / self.obss_std_tensor       
        return mem_obss.cpu().numpy()	      

    def step(
        self,
        obss: np.ndarray,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        self.model.eval()
        with torch.no_grad():
            inputs = np.concatenate([obss, actions], axis=-1)
            inputs = torch.from_numpy(inputs).float().to(self.device)
            mem_inputs, mem_targets = self.find_memories(inputs)
            
            obss_normalized = (obss - self.obss_mean) / self.obss_std
            actions_normalized = (actions - self.actions_mean) / self.actions_std
            inputs = np.concatenate([obss_normalized, actions_normalized], axis=-1)
            inputs = torch.from_numpy(inputs).float().to(self.device)
            
            dist = torch.norm(inputs - mem_inputs, dim=1).unsqueeze(1)
    

            preds = self.model(inputs=inputs, mem_targets=mem_targets, dist=dist, beta=0).cpu().numpy()

            # unnormalize predicted next states
            preds[..., :-1] = preds[..., :-1] * self.abs_max_delta_obss

            # undeltafy predicted next states
            next_obss = preds[:, :-1] + obss

            # unnormalize predicted rewards
            rewards = preds[:, -1:] * self.abs_max_rewards_diff
            rewards = rewards + self.rewards_mean

            terminals = self.terminal_fn(obss, actions, next_obss)
            info = {}
            info["raw_reward"] = rewards

            #if self.penalty_coef:
            if 0 > 1:
                mem_next_model = self.find_memories_penalty(torch.from_numpy(next_obss).float().to(self.device))
                delta_mem_obss_model = np.power(next_obss-mem_next_model, 2)
                penalty = np.sqrt(np.einsum('ij -> i', delta_mem_obss_model))
                penalty = -1 * penalty.reshape(10000, 1)

                # print(f'after clip {penalty.min()=} {penalty.max()=} \n')

                assert penalty.shape == rewards.shape
                rewards = rewards + self.penalty_coef * penalty
                info["penalty"] = penalty

            return next_obss, rewards, terminals, info
    
    def format_dataset_for_training(self, data: Dict, full_dataset: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        mem_obss = data["mem_observations"]
        mem_actions = data["mem_actions"]
        mem_next_obss = data["mem_next_observations"]
        mem_rewards = data["mem_rewards"]

        print(f'obss: {obss.shape}, actions: {actions.shape}, next_obss: {next_obss.shape}, rewards: {rewards.shape}, mem_obss: {mem_obss.shape}, mem_actions: {mem_actions.shape}, mem_next_obss: {mem_next_obss.shape}, mem_rewards: {mem_rewards.shape}')
        
        """
        self.obss_mean = np.mean(data["observations"], axis=0, keepdims=True)
        self.obss_std = np.std(data["observations"], axis=0, keepdims=True)
        self.obss_mean_tensor = torch.from_numpy(self.obss_mean).to(self.device)
        self.obss_std_tensor = torch.from_numpy(self.obss_std).to(self.device)
        
        self.actions_mean = np.mean(data["actions"], axis=0, keepdims=True)
        self.actions_std = np.std(data["actions"], axis=0, keepdims=True)
        self.actions_mean_tensor = torch.from_numpy(self.actions_mean).to(self.device)
        self.actions_std_tensor = torch.from_numpy(self.actions_std)
        """
        
        self.rewards_std = np.std(data["rewards"], axis=0, keepdims=True)

        delta_obss = (next_obss - obss) 
        delta_obss = delta_obss / self.abs_max_delta_obss
        delta_mem_obss = (mem_next_obss - mem_obss) / self.abs_max_delta_obss

        obss = (obss - self.obss_mean) / self.obss_std
        next_obss = (next_obss - self.obss_mean) / self.obss_std
        # actions = (actions - self.actions_mean) / self.actions_std

        mem_obss = (mem_obss - self.obss_mean) / self.obss_std
        mem_next_obss = (mem_next_obss - self.obss_mean) / self.obss_std
        # mem_actions = (mem_actions - self.actions_mean) / self.actions_std
        
        rewards = (rewards - self.rewards_mean) / self.abs_max_rewards_diff
        mem_rewards = (mem_rewards - self.rewards_mean) / self.abs_max_rewards_diff
        
        inputs = np.concatenate((obss, actions), axis=-1)
        mem_inputs = np.concatenate((mem_obss, mem_actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        mem_targets = np.concatenate((delta_mem_obss, mem_rewards), axis=-1)
        
        return inputs, targets, mem_inputs, mem_targets

    def train(
        self,
        data: Dict,
        full_dataset: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        use_tqdm: bool = False,
    ) -> None:
        self.use_tqdm = use_tqdm
        train_inputs, train_targets, train_mem_inputs, train_mem_targets = self.format_dataset_for_training(data, full_dataset)
                
        trainset = MemDataset(train_inputs, train_targets, train_mem_inputs, train_mem_targets)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        
        epoch = 0
        holdout_loss = 1e10
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            train_loss = self.train_model_epoch(epoch, trainloader)
                
            self.save(epoch, logger.model_dir)
            
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.set_timestep(epoch)
            logger.logkv("beta", self.beta)
            logger.dumpkvs(exclude=["policy_training_progress"])
            
            
            if (max_epochs and (epoch >= max_epochs)): 
                break        

    def train_model_epoch(self, epoch, trainloader):
        t0 = time.time()

        train_losses = []
        self.model.train()
        loader = tqdm(trainloader) if self.use_tqdm else trainloader
        for batch_idx, batch in enumerate(loader):
    
            step = batch_idx if epoch == 1 else (epoch-1)*self.num_batches + batch_idx
            self.beta = 0
            if self.len_of_beta_step != 0 and self.num_beta_steps != 0:
                self.beta = max(0.99 * (1 - int(step / self.len_of_beta_step)/self.num_beta_steps), 0)
                #print(f"beta:{self.beta}")
            
            train_loss = self.train_model_step(batch)
            #print(train_loss)
            
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

            self.diagnostics["beta/train"] = self.beta
        self.num_batches = batch_idx + 1
        return np.mean(train_losses)

    def train_model_step(self, batch):    
        new_batch = [item.to(self.device) for item in batch]
        inputs, targets, mem_inputs, mem_targets = new_batch

        with torch.no_grad():
            dist = self.compute_distances(inputs, mem_inputs)

        # print('inputs', inputs.max(), inputs.min(), inputs.mean())
        # print('mem_inputs', mem_inputs.max(), mem_inputs.min(), mem_inputs.mean())
        # print('dist', dist.max(), dist.min(), dist.mean())
        # print('targets', targets.max(), targets.min(), targets.mean())
        # print('mem_targets', mem_targets.max(), mem_targets.min(), mem_targets.mean())
        # print('beta', self.beta)

        preds = self.model(
            inputs=inputs,
            mem_targets=mem_targets,
            dist=dist,
            beta=self.beta, 
        )

        loss = F.mse_loss(preds, targets) 
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def test_model(self, testloader):
        self.model.eval()
        test_losses = []
        with torch.no_grad():
            loader = tqdm(testloader) if self.use_tqdm else testloader
            for batch in loader:
                test_loss = self.test_model_step(batch)
                test_losses.append(test_loss)
        return np.mean(test_losses)

    def test_model_step(self, batch):
        new_batch = [item.to(self.device) for item in batch]
        inputs, targets, mem_inputs, mem_targets = new_batch

        dist = self.compute_distances(inputs, mem_inputs)

        preds = self.model(
            inputs=inputs,
            mem_targets=mem_targets,
            dist=dist,
            beta=self.beta, 
        )

        # unnormalize predicted next states
        preds[..., :-1] = preds[..., :-1] * self.obss_abs_max_tensor


        loss = F.mse_loss(preds, targets) 

        return loss.detach().cpu().item()
    
    def compute_distances(self, inputs, mem_inputs, dim=1, pnorm=2):
        dist = torch.norm(inputs - mem_inputs, p=pnorm, dim=dim).unsqueeze(1)

        mean_dists = torch.mean(dist); max_dists = torch.max(dist); min_dists = torch.min(dist)
        self.diagnostics[f'dist/train/mean'] = mean_dists.item()
        self.diagnostics[f'dist/train/max'] = max_dists.item()
        self.diagnostics[f'dist/train/min'] = min_dists.item()
            
        return dist


    def save(self, epoch, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
        if epoch in [50, 100, 150]:
            torch.save(self.model.state_dict(), os.path.join(save_path, f"dynamics_{epoch}.pth"))

    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        #self.scaler.load_scaler(load_path)

        