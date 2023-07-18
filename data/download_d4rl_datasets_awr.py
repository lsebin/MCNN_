import gym
import numpy as np

import collections
import pickle

import d4rl
import os

# for more info on d4rl, see https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/infos.py

def download(name):
	print(name)
	env = gym.make(name)
	dataset = d4rl.qlearning_dataset(env) # env.get_dataset()

	N = dataset['rewards'].shape[0]
	data_ = collections.defaultdict(list)

	use_timeouts = False
	if 'timeouts' in dataset:
		use_timeouts = True

	episode_step = 0
	total_sum_rewards = []
	for i in range(N):
		total_sum=0
		done_bool = bool(dataset['terminals'][i])
		if use_timeouts:
			final_timestep = dataset['timeouts'][i]
		else:
			final_timestep = (episode_step == 1000-1)
   
		if not (bool(dataset['terminals'][i]) or (dataset['timeouts'][i] if use_timeouts else (episode_step == 1000-1))):
			total_sum += dataset['rewards']
			episode_step += 1
		else:
			total_sum_rewards.append(total_sum)
			total_sum = 0
			episode_step = 0
	
	print(f"Saved sums and episode num :{len(total_sum_rewards)}")
 
	episode_step = 0
	paths = []
	for i in range(N):
		episode_num = 0
		sum_until = 0
		done_bool = bool(dataset['terminals'][i])
		if use_timeouts:
			final_timestep = dataset['timeouts'][i]
		else:
			final_timestep = (episode_step == 1000-1)

		for k in ['observations', 'next_observations', 'actions', 'rewards' ,'terminals']:
			data_[k].append(dataset[k][i])
			data_['sum_rewards'].append(total_sum_rewards[episode_num]-sum_until)
			sum_until += dataset['rewards'][i]
		if done_bool or final_timestep:
			episode_step = 0
			episode_data = {}
			for k in data_:
				episode_data[k] = np.array(data_[k])
			paths.append(episode_data)
			data_ = collections.defaultdict(list)
			episode_num += 1
		else:
			episode_step += 1

	returns = np.array([np.sum(p['rewards']) for p in paths])
	num_samples = np.sum([p['rewards'].shape[0] for p in paths])
	print(f'Number of samples collected: {num_samples}')
	print(f'Number of episodes: {len(returns)}')
	print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

	with open(f'data/datasets_sum/{name}.pkl', 'wb') as f:
		pickle.dump(paths, f)


os.makedirs('data/datasets_sum', exist_ok=True)
# for env_name in ['halfcheetah', 'hopper', 'walker2d']:
#  	for dataset_type in ['random', 'medium', 'medium-replay', 'expert', 'medium-expert']:
#  		name = f'{env_name}-{dataset_type}-v2'
#  		download(name)

# for env_name in ['hammer', 'pen', 'relocate', 'door']:
# 	for dataset_type in ['human', 'expert', 'cloned']:
# 		name = f'{env_name}-{dataset_type}-v1'
# 		download(name)

# for name in ['carla-lane-v0', 'carla-town-v0', 'carla-town-full-v0']:
#	download(name)

"""
for dataset_type in ['random', 'medium', 'medium-replay', 'expert', 'medium-expert', 'random-expert]:
	name = f'ant-{dataset_type}-v0'
	download(name)
"""

# upgrade to numpy = 1.24... and try running it again

   
for env_name in ['antmaze-umaze']:#['antmaze-umaze', 'antmaze-medium', 'antmaze-large']
	if env_name == 'antmaze-umaze':
		for dataset_type in ['-diverse', '']:
			name = f'{env_name}{dataset_type}-v0'
			download(name)
	else :
		for dataset_type in ['diverse', 'play']:
			name = f'{env_name}-{dataset_type}-v0'
			download(name)

    	 
				
             