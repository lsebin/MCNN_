import argparse
import random

import gym
import d4rl

import os
import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import Critic, MemActor, TanhDiagGaussian, ActorProb
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.load_dataset import qlearning_dataset_percentbc
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs_td3bc
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import TD3BCPolicy, MemTD3BCPolicy
from offlinerlkit.modules import MemDynamicsModel
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.dynamics import MemDynamics
from offlinerlkit.policy import MNMPolicy
from offlinerlkit.policy_trainer import MBPolicyTrainer

"""
suggested hypers

halfcheetah-medium-v2: rollout-length=5, 
hopper-medium-v2: rollout-length=5, 
walker2d-medium-v2: rollout-length=5, 
halfcheetah-medium-replay-v2: rollout-length=5, 
hopper-medium-replay-v2: rollout-length=5, 
walker2d-medium-replay-v2: rollout-length=1, 
halfcheetah-medium-expert-v2: rollout-length=5, 
hopper-medium-expert-v2: rollout-length=5, 
walker2d-medium-expert-v2: rollout-length=1, 
"""

def get_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument("--algo-name", type=str, default="mem_mnm", choices=["mnm", "mem_mnm"])
    parser.add_argument("--task", type=str, default="halfcheetah-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # from mopo
    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--dynamics-epochs", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--model-retain-epochs", type=int, default=5)

    # tune
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=10000)
    
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--penalty-coef", type=float, default=0.5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)

    # for memories
    parser.add_argument('--chosen-percentage', type=float, default=1.0, choices=[0.1, 0.2, 0.5, 1.0])
    parser.add_argument('--num_memories_frac', type=float, default=0.1)
    parser.add_argument('--Lipz', type=float, default=1.0)
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--use-tqdm', type=int, default=1) # 1 or 0

    return parser.parse_args()


def train(args=get_args()):
    env = gym.make(args.task)
    dataset = qlearning_dataset_percentbc(args.task, args.chosen_percentage, args.num_memories_frac)
    if 'antmaze' in args.task:
        dataset["rewards"] -= 1.0
    args.obs_shape = (512,) if 'carla' in args.task else env.observation_space.shape
    args.obs_dim = np.prod(args.obs_shape)# create env and dataset
    
    args.action_dim = 2 if 'carla' in args.task else np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)
    fake_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    # scaler for normalizing observations
    scaler = StandardScaler(device=args.device)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # load perception encoder if carla
    if 'carla' in args.task:
        from offlinerlkit.carla.carla_model import CoILICRA
        from offlinerlkit.carla.carla_config import MODEL_CONFIGURATION
        carla_model_state_dict = torch.load('data/models/nocrash/resnet34imnet10S1/checkpoints/660000.pth')['state_dict']
        carla_model = CoILICRA(MODEL_CONFIGURATION)
        carla_model.load_state_dict(carla_model_state_dict)
        carla_model.eval()
        def perception_model(obs):
            obs = obs.reshape(1, 48, 48, 3)
            obs = torch.tensor(obs).permute(0, 3, 1, 2).float()
            return carla_model.perception(obs)[0].detach().numpy()
        print(f'loaded carla_model')
    else:
        perception_model = None

    # create dynamics model
    dynamics_model = MemDynamicsModel(
        input_dim=args.obs_dim + args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        output_dim=args.obs_dim + 1, # + 1 for reward
        Lipz=args.Lipz,
        lamda=args.lamda,
        device=args.device
    )
    dynamics_optim = torch.optim.AdamW(
        dynamics_model.parameters(),
        lr=args.dynamics_lr,
        weight_decay=args.dynamics_weight_decay[1],
    )
    
    # create policy model
    actor_hidden_dims = [1024, 1024] if 'carla' in args.task else [256, 256]
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=actor_hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=[256, 256])
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    
    # LATER ON, ADD THIS BACK! CAUSE IT WOULD BE GREAT TO HAVE MCNN POLICY AND MODEL!
    # Probably will have to add dist attribute to MemActor
    # if "mem" in args.algo_name: 
    #     actor = MemActor(actor_backbone, args.action_dim, device=args.device, Lipz=args.Lipz, lamda=args.lamda)
    # else:
    #     actor = ActorProb(actor_backbone, dist, args.device)
    
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    # create termination_fn and dynamics
    termination_fn = get_termination_fn(task=args.task)
    dynamics = MemDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn,
        dataset=dataset,
        penalty_coef=args.penalty_coef,
        # train_horizon=args.dynamics_train_horizon,
    )
    
    # log
    record_params = ["chosen_percentage", "rollout_length"]
    if "mem" in args.algo_name:
        record_params += ["num_memories_frac", "Lipz", "lamda"]

    log_dirs = make_log_dirs_td3bc(task_name=args.task, chosen_percentage=args.chosen_percentage, algo_name=args.algo_name, seed=args.seed, args=vars(args), record_params=record_params)
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))
    
    
    
    # train + save dynamics
    #dynamics.train(real_buffer.sample_all(), dataset, logger, max_epochs=args.dynamics_epochs, use_tqdm=args.use_tqdm)
    
    dynamics_path = f'{logger.model_dir}'
    if not os.path.exists(f'{dynamics_path}'):
        # dynamics.train(real_buffer.sample_all(), dataset, logger, max_epochs=args.dynamics_epochs, use_tqdm=args.use_tqdm)
        print(f'finished training dynamics and exiting. Uncomment exit here to go to train policy after everything worked fine with dynamics.')
    else:
        print(f'dynamics already exists. will be loaded from {dynamics_path}')
        dynamics.load(f'{dynamics_path}')
        
    # create policy 
    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha
    
    policy = MNMPolicy(
        dynamics,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha
    )

    # create trainer(s)
    policy_trainer = MBPolicyTrainer(
        policy=policy,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler
    )

    # train policy
    policy_trainer.train(use_tqdm=args.use_tqdm)


if __name__ == "__main__":
    train()