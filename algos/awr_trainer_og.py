import argparse
import random
from collections import deque

import gym
import d4rl
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from offlinerlkit.buffer import ReplayBuffer
from torch.multiprocessing import Process
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.load_dataset import qlearning_dataset_percentbc_awr

from offlinerlkit.nets.awr_model_og import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="awr", choices=["awr", "mem_awr"])
    parser.add_argument("--task", type=str, default='antmaze-umaze-v0')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=2.5) # reassign to new hyper for AWR
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    parser.add_argument('--chosen-percentage', type=float, default=1.0, choices=[0.1, 0.2, 0.5, 1.0])
    parser.add_argument('--num_memories_frac', type=float, default=0.1)
    parser.add_argument('--Lipz', type=float, default=1.0)
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--use-tqdm', type=int, default=1) # 1 or 0

    return parser.parse_args()


class RLEnv(Process):
    def __init__(self, env_id, is_render):

        super(RLEnv, self).__init__()

        self.daemon = True
        self.env = gym.make(env_id)

        self.is_render = is_render
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.recent_rlist.append(0)

        self.reset()

    def step(self, action):
        if self.is_render:
            self.env.render()

        obs, reward, done, info = self.env.step(action)

        self.rall += reward
        self.steps += 1

        if done:
            if self.steps < self.env.spec.max_episode_steps:
                reward = -1

            self.recent_rlist.append(self.rall)
            print("[Episode {}] Reward: {}  Recent Reward: {}".format(
                self.episode, self.rall, np.mean(self.recent_rlist)))
            obs = self.reset()

        return obs, reward, done, info

    def reset(self):
        self.steps = 0
        self.episode += 1
        self.rall = 0

        return np.array(self.env.reset())


class ActorAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            gamma,
            dataset,
            action_dim,
            rewards_dim,
            Lipz,
            lamda,
            device,
            scaler,
            lam=0.95,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            use_continuous=False):
        self.model = BaseActorCriticNetwork(
            input_size, output_size, action_dim, rewards_dim, Lipz, lamda, device,
            use_noisy_net=use_noisy_net, use_continuous=use_continuous)
        self.continuous_agent = use_continuous
        
        self.output_size = output_size
        self.input_size = input_size
        self.gamma = gamma
        self.lam = lam
        self.use_gae = use_gae

        self.actor_optimizer = optim.SGD(self.model.actor.parameters(),
                                          lr=0.00005, momentum=0.9)
        self.critic_optimizer = optim.SGD(self.model.critic.parameters(),
                                           lr=0.0001, momentum=0.9)
        self.device=device
        self.model = self.model.to(self.device)
        
        dataset["memories_obs"] = scaler.transform(dataset["memories_obs"])
        dataset["memories_next_obs"] = scaler.transform(dataset["memories_next_obs"])
        
        self.nodes_obs = torch.from_numpy(dataset["memories_obs"]).float().to(self.device)
        self.nodes_actions = torch.from_numpy(dataset["memories_actions"]).float().to(self.device)
        self.nodes_next_obs = torch.from_numpy(dataset["memories_next_obs"]).float().to(self.device)
        self.nodes_rewards = torch.from_numpy(dataset["memories_rewards"]).float().to(self.device).unsqueeze(1)
        self.nodes_sum_rewards = torch.from_numpy(dataset["memories_sum_rewards"]).float().to(self.device).unsqueeze(1)
        

    def train_model(self, s_batch, action_batch, reward_batch, n_s_batch, done_batch):
        data_len = len(np.array(s_batch.cpu()))
        mse = nn.MSELoss()
        
        # find closest memory
        _, closest_nodes = torch.cdist(s_batch, self.nodes_obs).min(dim=1)
        mem_state = self.nodes_obs[closest_nodes, :]
        mem_action = self.nodes_actions[closest_nodes, :]
        mem_sum_rewards = self.nodes_sum_rewards[closest_nodes, :]
        dist = torch.norm(s_batch - mem_state, p=2, dim=1).unsqueeze(1)
        print(f's_batch:{s_batch.shape}, mem_state={mem_state.shape}, mem_Action={mem_action.shape}, mem_sum_rewards={mem_sum_rewards.shape}, dist={dist.shape}')
        
        s_batch = np.array(s_batch.cpu())
        action_batch = np.array(action_batch.cpu())
        reward_batch = np.array(reward_batch.cpu())
        done_batch = np.array(done_batch.cpu())
        
        #update critic
        self.critic_optimizer.zero_grad()
        cur_value = self.model.critic(s_batch, mem_sum_rewards, dist, 0)
        print(f'cur_value:{cur_value.shape}')
        print('Before opt - Value has nan: {}'.format(torch.sum(torch.isnan(cur_value))))
        discounted_reward, _ = discount_return(reward_batch, done_batch, cur_value.cpu().detach().numpy())
        # discounted_reward = (discounted_reward - discounted_reward.mean())/(discounted_reward.std() + 1e-8)
        for _ in range(critic_update_iter):
            sample_idx = random.sample(range(data_len), 256)
            sample_value = self.model.critic(s_batch[sample_idx], mem_sum_rewards[sample_idx], dist[sample_idx], beta=0)
            if (torch.sum(torch.isnan(sample_value)) > 0):
                print('NaN in value prediction')
                input()
            critic_loss = mse(sample_value.squeeze(), torch.FloatTensor(discounted_reward[sample_idx]).to(args.device))
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()

        # update actor
        cur_value = self.model.critic(s_batch, mem_sum_rewards, dist, beta=0)
        print('After opt - Value has nan: {}'.format(torch.sum(torch.isnan(cur_value))))
        discounted_reward, adv = discount_return(reward_batch, done_batch, cur_value.cpu().detach().numpy())
        print('Advantage has nan: {}'.format(torch.sum(torch.isnan(torch.tensor(adv).float()))))
        print('Returns has nan: {}'.format(torch.sum(torch.isnan(torch.tensor(discounted_reward).float()))))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.actor_optimizer.zero_grad()
        for _ in range(actor_update_iter):
            sample_idx = random.sample(range(data_len), 256)
            weight = torch.tensor(np.minimum(np.exp(adv[sample_idx] / beta), max_weight)).float().reshape(-1, 1)
            #print(s_batch[sample_idx].type, mem_action[sample_idx].type, dist[sample_idx].type)
            cur_policy = self.model.actor(s_batch[sample_idx], mem_action[sample_idx], dist[sample_idx], beta=0)
            if self.continuous_agent:
                actor_loss = mse(cur_policy.squeeze(), torch.FloatTensor(action_batch[sample_idx]).to(args.device))
            else:
                m = Categorical(F.softmax(cur_policy[:, :, None], dim=-1))
                actor_loss = -m.log_prob(torch.LongTensor(action_batch[sample_idx])) * weight.reshape(-1)
                actor_loss = actor_loss.mean()
            # print(actor_loss)

            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()

        print('Weight has nan {}'.format(torch.sum(torch.isnan(weight))))


def discount_return(reward, done, value):
    value = value.squeeze()
    num_step = len(value)
    discounted_return = np.zeros([num_step])
    gae = 0
    for t in range(num_step - 1, -1, -1):
        if done[t] or t == num_step - 1:
        #if done[t]:
            delta = reward[t] - value[t]
        else:
            delta = reward[t] + gamma * value[t + 1] - value[t]
        #print(reward[t], value[t], delta, gamma, done[t], lam, gae)
        gae = delta + gamma * lam * (1 - done[t]) * gae

        discounted_return[t] = gae + value[t]

    # For Actor
    adv = discounted_return - value
    return discounted_return, adv


if __name__ == '__main__':
    args = get_args()
    
    print(args.task)
    env = gym.make(args.task)

    continuous = isinstance(env.action_space, gym.spaces.Box)
    print('Env is continuous: {}'.format(continuous))
    
    args.action_dim = np.prod(env.action_space.shape)
    rewards_dim = 1

    input_size = env.observation_space.shape[0]  # 4
    output_size = env.action_space.shape[0] if continuous else env.action_space.n  # 2
    env.close()
    
    is_awr = True
    dataset = qlearning_dataset_percentbc_awr(args.task, args.chosen_percentage, args.num_memories_frac, is_awr)
    if 'antmaze' in args.task:
        dataset["rewards"] -= 1.0
        
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=env.observation_space.shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)
    obs_mean, obs_std = buffer.normalize_obs()
    scaler = StandardScaler(mu=obs_mean, std=obs_std)
    
    # normalize where?? -> buffer is already normalized here but dataset isn't normalized

    use_cuda = False
    use_noisy_net = False
    num_sample = 2048
    critic_update_iter = 500
    actor_update_iter = 1000
    iteration = 100000
    max_replay = 50000
    
    gamma = args.gamma
    lam = 0.95
    beta = 0.05
    max_weight = 20.0
    use_gae = True

    agent = ActorAgent(
        input_size,
        output_size,
        args.gamma,
        dataset,
        args.action_dim,
        rewards_dim,
        args.Lipz,
        args.lamda,
        args.device,
        scaler,
        use_gae=use_gae,
        use_cuda=use_cuda,
        use_noisy_net=use_noisy_net,
        use_continuous=continuous,
        )

    last_done_index = -1

    for i in range(iteration):
        batch = buffer.sample(args.batch_size)
        states, actions, rewards, next_states, dones = batch['observations'], batch['actions'], batch['rewards'], batch['next_observations'], batch["terminals"]
        agent.train_model(states, actions, rewards, next_states, dones)