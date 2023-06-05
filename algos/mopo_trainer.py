import argparse
import os
import sys
import random

import gym
import d4rl

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel, MemDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics, MemEnsembleDynamics, MemDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
#from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.utils.load_dataset import qlearning_dataset_percentbc
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MBPolicyTrainer
from offlinerlkit.policy import MOPOPolicy
from offlinerlkit.planners import mppi


"""
suggested hypers

halfcheetah-medium-v2: rollout-length=5, penalty-coef=0.5
hopper-medium-v2: rollout-length=5, penalty-coef=5.0
walker2d-medium-v2: rollout-length=5, penalty-coef=0.5
halfcheetah-medium-replay-v2: rollout-length=5, penalty-coef=0.5
hopper-medium-replay-v2: rollout-length=5, penalty-coef=2.5
walker2d-medium-replay-v2: rollout-length=1, penalty-coef=2.5
halfcheetah-medium-expert-v2: rollout-length=5, penalty-coef=2.5
hopper-medium-expert-v2: rollout-length=5, penalty-coef=5.0
walker2d-medium-expert-v2: rollout-length=1, penalty-coef=2.5
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mem_mopo", choices=["mopo", "memensemble_mopo", "mem_mopo"])
    parser.add_argument("--task", type=str, default="halfcheetah-medium-replay-v2")
    parser.add_argument("--train-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--dynamics-epochs", type=int, default=200)
    # parser.add_argument("--dynamics-train-horizon", type=int, default=5)
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--penalty-coef", type=float, default=0.5)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--phase", type=str, default="train_dynamics", choices=["train_dynamics", "train_policy", "tune_mppi", "test_mppi"])

    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument('--num_memories_frac', type=float, default=0.1)
    parser.add_argument('--Lipz', type=float, default=15.0)
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--use_tqdm', type=int, default=1) # 1 or 0

    parser.add_argument("--mppi-horizon", type=int, default=5)
    
    # for memories
    parser.add_argument('--chosen-percentage', type=float, default=1.0, choices=[0.1, 0.2, 0.5, 1.0])

    return parser.parse_args()


def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    #dataset = qlearning_dataset(args.task, args.train_size, args.num_memories_frac)
    dataset = qlearning_dataset_percentbc(args.task, args.chosen_percentage, args.num_memories_frac)
    if 'antmaze' in args.task:
        dataset["rewards"] -= 1.0
    args.obs_shape = env.observation_space.shape
    args.obs_dim = np.prod(args.obs_shape)
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    if args.algo_name == "mopo" or args.algo_name == "memensemble_mopo":
        dynamics_model = EnsembleDynamicsModel(
            obs_dim=np.prod(args.obs_shape),
            action_dim=args.action_dim,
            hidden_dims=args.dynamics_hidden_dims,
            num_ensemble=args.n_ensemble,
            num_elites=args.n_elites,
            weight_decays=args.dynamics_weight_decay,
            device=args.device
        )
        dynamics_optim = torch.optim.Adam(
            dynamics_model.parameters(),
            lr=args.dynamics_lr
        )
    elif args.algo_name == "mem_mopo":
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
    
    scaler = StandardScaler(device=args.device)
    termination_fn = get_termination_fn(task=args.task)

    if args.algo_name == "mopo":
        dynamics = EnsembleDynamics(
            dynamics_model,
            dynamics_optim,
            scaler,
            termination_fn,
            penalty_coef=args.penalty_coef,
        )
    elif args.algo_name == "memensemble_mopo":
        dynamics = MemEnsembleDynamics(
            dynamics_model,
            dynamics_optim,
            scaler,
            termination_fn,
            penalty_coef=args.penalty_coef,
            dataset=dataset,
            Lipz=args.Lipz,
            lamda=args.lamda,
        )
    elif args.algo_name == "mem_mopo":
        dynamics = MemDynamics(
            dynamics_model,
            dynamics_optim,
            scaler,
            termination_fn,
            dataset=dataset,
            penalty_coef=args.penalty_coef,
            # train_horizon=args.dynamics_train_horizon,
        )

    if args.phase == "test_mppi":
        vanilla_dynamics_model = EnsembleDynamicsModel(
            obs_dim=np.prod(args.obs_shape),
            action_dim=args.action_dim,
            hidden_dims=args.dynamics_hidden_dims,
            num_ensemble=args.n_ensemble,
            num_elites=args.n_elites,
            weight_decays=args.dynamics_weight_decay,
            device=args.device
        )
        vanilla_dynamics_optim = torch.optim.Adam(
            dynamics_model.parameters(),
            lr=args.dynamics_lr
        )
        vanilla_dynamics = EnsembleDynamics(
            vanilla_dynamics_model,
            vanilla_dynamics_optim,
            scaler,
            termination_fn,
            penalty_coef=args.penalty_coef,
        )

    # log
    record_params = ["train_size", "penalty_coef", "rollout_length"]
    if args.algo_name == "memensemble_mopo" or args.algo_name == "mem_mopo":
        record_params += ["num_memories_frac", "Lipz", "lamda"]
    # if args.algo_name == "mem_mopo":
    #     record_params += ["dynamics_train_horizon"]
    log_dirs = make_log_dirs(args.task, args.train_size, args.algo_name, args.seed, vars(args), record_params=record_params)
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)

    # load dynamics if training policy
    if args.phase == "train_policy" or "mppi" in args.phase:
        # e.g. load_folder = "algos/exp/halfcheetah-medium-v2/{algo}&train_size=0.1&penalty_coef=0.5&rollout_length=5&num_memories_frac=0.05&Lipz=100.0&lamda=1.0/seed_1/model"
        load_folder = '&'.join(logger.model_dir.split('&')[:2]) + f'&penalty_coef=0.5&' + '&'.join(logger.model_dir.split('&')[3:])
        print(f'Loading dynamics from {load_folder}')
        dynamics.load(load_folder)

        if args.phase == "test_mppi":
            print(logger.model_dir.split('&'))
            vanilla_load_folder = '/'.join(logger.model_dir.split('&')[0].split('/')[:-1]) + f'/mopo&' + logger.model_dir.split('&')[1] + f'&penalty_coef=0.5&' + logger.model_dir.split('&')[3] + f'/' + '/'.join(logger.model_dir.split('&')[-1].split('/')[1:])
            print(f'Loading vanilla dynamics from {vanilla_load_folder}')
            vanilla_dynamics.load(vanilla_load_folder)

    # create policy
    policy = MOPOPolicy(
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

    # if args.phase == "tune_mppi":
    #     load_folder = '&'.join(logger.checkpoint_dir.split('&')[:2]) + f'&penalty_coef=0.5&' + '&'.join(logger.checkpoint_dir.split('&')[3:])
    #     policy.load_state_dict(torch.load(os.path.join(load_folder, "policy.pth")))

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

    # log hyperparams (some ints become floats in this process and hence the below line is here and not above where, for e.g., replay buffer needs args.action_dim to be int)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
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

    # train
    if args.phase == "train_dynamics":
        dynamics.train(real_buffer.sample_all(), dataset, logger, max_epochs=args.dynamics_epochs, use_tqdm=args.use_tqdm)
    elif args.phase == "train_policy":
        policy_trainer.train(use_tqdm=args.use_tqdm)
    elif args.phase == "tune_mppi":
        mppi.tune(env, dynamics, policy, args.device, args.mppi_horizon)
    elif args.phase == "test_mppi":
        mppi.test(env, dynamics, vanilla_dynamics, args.device)
    else:
        raise ValueError("Invalid args.phase")


if __name__ == "__main__":
    train()