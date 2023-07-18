import argparse
import random
from collections import deque

import gym
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from offlinerlkit.buffer import ReplayBuffer
from torch.multiprocessing import Process
from offlinerlkit.utils.load_dataset import qlearning_dataset_percentbc_awr

from offlinerlkit.nets.awr_model_og import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="awr", choices=["awr", "mem_awr"])
    parser.add_argument("--task", type=str, default="hopper-expert-v2")
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
    
    parser.add_argument('--chosen-percentage', type=float, default=0.1, choices=[0.1, 0.2, 0.5, 1.0])
    parser.add_argument('--num_memories_frac', type=float, default=0.1)
    parser.add_argument('--Lipz', type=float, default=15.0)
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

        # try: 
        # obs, reward, done, info = self.env.step(action)
        # except: 
        #     input(action)
        #     obs, reward, done, info = self.env.step(action)

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
            Lipz,
            lamda,
            device,
            lam=0.95,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            use_continuous=False):
        self.model = BaseActorCriticNetwork(
            input_size, output_size, action_dim, Lipz, lamda, device,
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
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.model.to(self.device)
        
        self.nodes_obs = torch.from_numpy(dataset["memories_obs"]).float().to(self.device)
        self.nodes_actions = torch.from_numpy(dataset["memories_actions"]).float().to(self.device)
        self.nodes_next_obs = torch.from_numpy(dataset["memories_next_obs"]).float().to(self.device)
        self.nodes_rewards = torch.from_numpy(dataset["memories_rewards"]).float().to(self.device).unsqueeze(1)
        self.nodes_sum_rewards = torch.from_numpy(dataset["memories_sum_rewards"]).float().to(self.device).unsqueeze(1)
        
        
    # def get_action(self, state):
    #     # state = torch.Tensor(state).to(self.device).reshape(1,-1)
    #     # state = state.float()
    #     state = torch.tensor(state).float().reshape(1, -1)
        
    #     # need to also give memory and memory target
    #     policy, value = self.model(state, mem_state, mem_action, mem_sum_rewards)

    #     if self.continuous_agent:
    #         action = policy.sample().numpy().reshape(-1)
    #     else:
    #         policy = F.softmax(policy, dim=-1).data.cpu().numpy()
    #         action = np.random.choice(np.arange(self.output_size), p=policy[0])

    #     return action

    def train_model(self, s_batch, action_batch, reward_batch, n_s_batch, done_batch):
        s_batch = np.array(s_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        done_batch = np.array(done_batch)

        data_len = len(s_batch)
        mse = nn.MSELoss()
        
        # find closest memory
        _, closest_nodes = torch.cdist(s_batch, self.nodes_inputs).min(dim=1)
        mem_state = self.nodes_obs[closest_nodes, :]
        mem_action = self.nodes_actions[closest_nodes, :]
        mem_sum_rewards = self.nodes_sum_rewards[closest_nodes, :]
        dist = torch.norm(s_batch - mem_state, p=2, dim=1).unsqueeze(1)

        # update critic
        self.critic_optimizer.zero_grad()
        cur_values = self.model.critic(s_batch, mem_state, mem_sum_rewards, dist, beta=0)
        #cur_value = self.model.critic(torch.FloatTensor(s_batch))
        print('Before opt - Value has nan: {}'.format(torch.sum(torch.isnan(cur_value))))
        discounted_reward, _ = discount_return(reward_batch, done_batch, cur_value.cpu().detach().numpy())
        # discounted_reward = (discounted_reward - discounted_reward.mean())/(discounted_reward.std() + 1e-8)
        for _ in range(critic_update_iter):
            sample_idx = random.sample(range(data_len), 256)
            # _, closest_nodes = torch.cdist(s_batch[sample_idx], self.nodes_inputs).min(dim=1)
            # mem_state = self.nodes_obs[closest_nodes, :]
            # mem_sum_rewards = self.nodes_sum_rewards[closest_nodes, :]
            # dist = torch.norm(s_batch[sample_idx] - mem_state, p=2, dim=1).unsqueeze(1)
            sample_value = self.model.critic(s_batch[sample_idx], mem_state[sample_idx], mem_sum_rewards[sample_idx], dist[sample_idx], beta=0)
            if (torch.sum(torch.isnan(sample_value)) > 0):
                print('NaN in value prediction')
                input()
            critic_loss = mse(sample_value.squeeze(), torch.FloatTensor(discounted_reward[sample_idx]))
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()

        # update actor
        cur_values = self.model.critic(s_batch, mem_state, mem_sum_rewards, dist, beta=0)
        #cur_value = self.model.critic(torch.FloatTensor(s_batch))
        print('After opt - Value has nan: {}'.format(torch.sum(torch.isnan(cur_value))))
        discounted_reward, adv = discount_return(reward_batch, done_batch, cur_value.cpu().detach().numpy())
        print('Advantage has nan: {}'.format(torch.sum(torch.isnan(torch.tensor(adv).float()))))
        print('Returns has nan: {}'.format(torch.sum(torch.isnan(torch.tensor(discounted_reward).float()))))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.actor_optimizer.zero_grad()
        for _ in range(actor_update_iter):
            sample_idx = random.sample(range(data_len), 256)
            weight = torch.tensor(np.minimum(np.exp(adv[sample_idx] / beta), max_weight)).float().reshape(-1, 1)
            # cur_policy = self.model.actor(torch.FloatTensor(s_batch[sample_idx]))
            #  _, closest_nodes = torch.cdist(s_batch[sample_idx], self.nodes_inputs).min(dim=1)
            # mem_state = self.nodes_obs[closest_nodes, :]
            # mem_action = self.nodes_actions[closest_nodes, :]
            # dist = torch.norm(s_batch[sample_idx] - mem_state, p=2, dim=1).unsqueeze(1)
            cur_policy = self.model.actor(s_batch[sample_idx], mem_state[sample_idx], mem_action[sample_idx], dist[sample_idx], beta=0)

            if self.continuous_agent:
                probs = -cur_policy.log_probs(torch.tensor(action_batch[sample_idx]).float())
                actor_loss = probs * weight
            else:
                m = Categorical(F.softmax(cur_policy, dim=-1))
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
        if done[t]:
            delta = reward[t] - value[t]
        else:
            delta = reward[t] + gamma * value[t + 1] - value[t]
        gae = delta + gamma * lam * (1 - done[t]) * gae

        discounted_return[t] = gae + value[t]

    # For Actor
    adv = discounted_return - value
    return discounted_return, adv


if __name__ == '__main__':
    args = get_args()
    # env_id = 'CartPole-v1'
    # env_id = 'Pendulum-v0'
    # env_id = 'LunarLanderContinuous-v2'
    # env_id = 'Acrobot-v1'
    # env_id = 'BipedalWalker-v2'

    env = gym.make(args.task)

    continuous = isinstance(env.action_space, gym.spaces.Box)
    print('Env is continuous: {}'.format(continuous))
    
    args.action_dim = np.prod(env.action_space.shape)

    input_size = env.observation_space.shape[0]  # 4
    output_size = env.action_space.shape[0] if continuous else env.action_space.n  # 2
    env.close()
    
    is_awr = True
    dataset = qlearning_dataset_percentbc_awr(args.task, args.chosen_percentage, args.num_memories_frac, is_awr)
    if 'antmaze' in args.task:
        dataset["rewards"] -= 1.0
        
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)
    obs_mean, obs_std = buffer.normalize_obs()

    use_cuda = False
    use_noisy_net = False
    batch_size = 256
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
        args.Lipz,
        args.lamda,
        args.device,
        use_gae=use_gae,
        use_cuda=use_cuda,
        use_noisy_net=use_noisy_net,
        use_continuous=continuous,
        )
    is_render = False

    #env = RLEnv(env_id, is_render)
    #env = RLEnv(args.task, is_render)
    
    # states, actions, rewards, next_states, dones = deque(maxlen=max_replay), deque(maxlen=max_replay), deque(
    #     maxlen=max_replay), deque(maxlen=max_replay), deque(maxlen=max_replay)

    last_done_index = -1

    for i in range(iteration):
        batch = buffer.sample(args.batch_size)
        states, actions, rewards, next_states, dones = batch['observations'], batch['actions'], batch['rewards'], batch['next_observations'], batch["terminals"]
        # Online RL part here
        # done = False
        # score = 0

        # step = 0
        # episode = 0
        # state = env.reset()

        # while True:
        #     step += 1
        #     action = agent.get_action(state)
        #     if (torch.sum(torch.isnan(torch.tensor(action).float()))):
        #         print(action)
        #         action = np.zeros_like(action)
        #     next_state, reward, done, info = env.step(action)
        #     states.append(np.array(state))
        #     actions.append(action)
        #     rewards.append(reward)
        #     next_states.append(np.array(next_state))
        #     dones.append(done)

        #     state = next_state[:]

        #     if done:
        #         episode += 1

        #         state = env.reset()
        #         if step > num_sample:
        #             step = 0
        #             # train
        agent.train_model(states, actions, rewards, next_states, dones)