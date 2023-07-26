import argparse
import random
from collections import deque
import time
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
from offlinerlkit.utils.logger import Logger, make_log_dirs_origin
from offlinerlkit.nets.awr_model_og import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mem_awr", choices=["awr", "mem_awr"])
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
    parser.add_argument('--critic-update-iter', type=int, default=500) # 1 or 0
    parser.add_argument('--actor-update-iter', type=int, default=1000)
    parser.add_argument('--iteration', type=int, default=2000)

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
            buffer,
            action_dim,
            scaler,
            batch_size,
            device,
            lam=0.95,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            use_continuous=False):
        self.model = BaseActorCriticNetwork_og(
            input_size, action_dim, use_noisy_net=use_noisy_net, use_continuous=use_continuous)
        self.continuous_agent = use_continuous
        
        self.output_size = output_size
        self.input_size = input_size
        self.gamma = gamma
        self.lam = lam
        self.use_gae = use_gae
        self.batch_size = batch_size

        self.actor_optimizer = optim.SGD(self.model.actor.parameters(),
                                          lr=0.00005, momentum=0.9)
        self.critic_optimizer = optim.SGD(self.model.critic.parameters(),
                                           lr=0.0001, momentum=0.9)
        self.device=device
        self.model = self.model.to(self.device)

    def train_model(self, buffer, mean_sum_rewards, abs_max_sum_rewards) :
        self.model.actor.train()
        self.model.critic.train()
        
        mse_critic = nn.MSELoss()
        mse_actor = nn.MSELoss()
                        
        result={}
        actor_losses = []
        critic_losses = []

        # update critic
        self.critic_optimizer.zero_grad()
        critic_start_time = time.time()
        for iter in range(args.critic_update_iter):
            # sample batch from buffer
            batch = buffer.sample(args.batch_size)

            # get data from batch including discounted rewards, compute dist
            states, actions, rewards = batch['observations'], batch['actions'], batch['rewards']
            discounted_return = batch['sum_rewards']

            # forward pass on critic network
            sample_value =  self.model.critic(states)
            sample_value = sample_value * abs_max_sum_rewards + mean_sum_rewards
            
            if (torch.sum(torch.isnan(sample_value)) > 0):
                print('NaN in value prediction')
                input()
            
            # compute loss and backward pass
            critic_loss = mse_critic(sample_value.squeeze(), discounted_return) 
            critic_losses.append(critic_loss.item())
            critic_loss.backward() 
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()
            
        result.update({"loss/critic": np.mean(critic_losses),})
        print("critic time: {:.2f}s".format(time.time() - critic_start_time))

        # update actor
        self.actor_optimizer.zero_grad()
        actor_start_time = time.time()
        for iter in range(args.actor_update_iter):
            # sample batch from buffer
            batch = buffer.sample(args.batch_size)

            # get data from batch including discounted rewards, compute dist
            states, actions, rewards, dones = batch['observations'], batch['actions'], batch['rewards'], batch["terminals"]
            discounted_return = batch['sum_rewards']

            # compute advantage to use as weights for "advantge weighted" regression
            value = self.model.critic(states).detach().clone()
            adv = discounted_return - value # no gradients here
            weight = torch.minimum(torch.exp(adv / beta), max_weight).reshape(-1, 1)
            
            # forward pass on actor network
            cur_policy = self.model.actor(states)
            
            # compute loss and backward pass
            if self.continuous_agent:
                actor_loss = (weight * (cur_policy - actions).pow(2))
            else:
                m = Categorical(F.softmax(cur_policy, dim=-1))
                actor_loss = -m.log_prob(torch.LongTensor(actions)) * weight.reshape(-1)
                
            actor_loss = actor_loss.mean()
            actor_losses.append(actor_loss.item())

            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()
        
        result.update({"loss/actor": np.mean(actor_losses),})
        
        print("actor time: {:.2f}s".format(time.time() - actor_start_time))
        print('Weight has nan {}'.format(torch.sum(torch.isnan(weight))))
        
        # maybe change it to mean later
        return result
        
    def evaluate_model(self, eval_env):
        self.model.actor.eval()
    
        obs = eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        
        while num_episodes < args.eval_episodes:
            obs = scaler.transform(obs)
            obs = torch.from_numpy(obs).float().to(self.device)
            action = self.model.actor(obs)
            
            action = action.cpu().detach().numpy()
            next_obs, reward, terminal, _ = eval_env.step(action.flatten())
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = eval_env.reset()
        
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }


if __name__ == '__main__':
    args = get_args()
    
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    print(args.task)
    env = gym.make(args.task)
    env.seed(args.seed)

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
        device=args.device,
        is_awr = is_awr
    )
    buffer.load_dataset(dataset)
    obs_mean, obs_std = buffer.normalize_obs()
    
    scaler = StandardScaler(mu=obs_mean, std=obs_std)
    
    mean_sum_rewards = torch.tensor(dataset['mean_sum_rewards']).to(args.device)
    abs_max_sum_rewards = torch.tensor(dataset['abs_max_sum_rewards']).to(args.device)

    use_cuda = True if args.device == "cuda" else False #False
    use_noisy_net = False
    num_sample = 2048
    
    gamma = args.gamma
    lam = 0.95
    beta = 0.05
    max_weight = torch.FloatTensor(np.full((args.batch_size, 1), 20.0)).to(args.device)
    use_gae = True

    agent = ActorAgent(
        input_size,
        output_size,
        args.gamma,
        buffer,
        args.action_dim,
        scaler,
        args.batch_size,
        args.device,
        args.batch_size,
        use_gae=use_gae,
        use_cuda=use_cuda,
        use_noisy_net=use_noisy_net,
        use_continuous=continuous,
        )

    last_done_index = -1
    
    log_dirs=make_log_dirs_origin(args.task, args.algo_name, args.seed, args.Lipz, args.lamda, vars(args), args.num_memories_frac)
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    num_paths = args.batch_size
    print(num_paths)
    
    last_10_performance = deque(maxlen=10)
    start_time = time.time()

    for i in range(args.iteration):
        loop_start_time = time.time()
        
        loss = agent.train_model(buffer, mean_sum_rewards, abs_max_sum_rewards)
        
        if i % 50 == 0:
            eval_info = agent.evaluate_model(env)
        
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            norm_ep_rew_mean = env.get_normalized_score(ep_reward_mean) * 100
            norm_ep_rew_std = env.get_normalized_score(ep_reward_std) * 100
            last_10_performance.append(norm_ep_rew_mean)
            logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
            logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
            logger.logkv("eval/episode_length", ep_length_mean)
            logger.logkv("eval/episode_length_std", ep_length_std)
            
        logger.set_timestep(i)
        for k, v in loss.items():
            logger.logkv_mean(k, v)
        logger.dumpkvs()
        
        print(f"epoch {i} time: {time.time() - loop_start_time}s")
      
    logger.log("total time: {:.2f}s".format(time.time() - start_time))
    logger.close()
    print(f'last_10_performance : {np.mean(last_10_performance)}')