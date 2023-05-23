import argparse
import gym  
import d4rl

ibc_data = { # https://arxiv.org/pdf/2109.00137.pdf
    'pen-human-v1': (2446, 207),
    'hammer-human-v1': (-9.3, 45.5),
    'door-human-v0': (399, 34),
    'relocate-human-v0': (3.6, 2.5),
}
for task, (mean, std) in ibc_data.items():
    env = gym.make(task)
    print(env.observation_space.shape)
    print(f'\n', task, f'mean: ', env.get_normalized_score(mean) * 100, f'std: ', env.get_normalized_score(std) * 100, f'\n', )
