import gym
import numpy as np

import collections
import pickle

import d4rl
import os
import argparse 

# for more info on d4rl, see https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/infos.py


# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--is_awr', default=False, type=bool)
parser.add_argument('--name', type=str)


def download(name, is_awr):
	print(name)
	env = gym.make(name)
	dataset = d4rl.qlearning_dataset(env)

	N = dataset['rewards'].shape[0]
	data_ = collections.defaultdict(list)

	use_timeouts = False
	if 'timeouts' in dataset:
		use_timeouts = True
  
	if is_awr: 
		episode_step = 0
		total_sum_rewards = []
		total_sum = 0
		for i in range(N):
			done_bool = bool(dataset['terminals'][i])
			if use_timeouts:
				final_timestep = dataset['timeouts'][i]
			else:
				final_timestep = (episode_step == 1000-1)
    
			total_sum += dataset['rewards'][i]
   
			if done_bool or final_timestep:
				total_sum_rewards.append(total_sum)
				total_sum = 0
				episode_step = 0
			else:
				episode_step += 1
				
		total_sum_rewards.append(total_sum)
		
		print(f"Saved sums and episode num: {len(total_sum_rewards)}")
 
	episode_step = 0
	paths = []
	sum_until = 0
	episode_num = 0
	for i in range(N):
		done_bool = bool(dataset['terminals'][i])
		if use_timeouts:
			final_timestep = dataset['timeouts'][i]
		else:
			final_timestep = (episode_step == 1000-1)

		for k in ['observations', 'next_observations', 'actions', 'rewards' ,'terminals']:
			data_[k].append(dataset[k][i])
		if is_awr:
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
			sum_until = 0
		else:
			episode_step += 1
   
	print(episode_num)

	returns = np.array([np.sum(p['rewards']) for p in paths])
	num_samples = np.sum([p['rewards'].shape[0] for p in paths])
	print(f'Number of samples collected: {num_samples}')
	print(f'Number of episodes: {len(returns)}')
	print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

	with open(f'data/datasets_sum/{name}.pkl', 'wb') as f:
		pickle.dump(paths, f)


args = parser.parse_args()
savepkl = 'data/datasets_sum' if args.is_awr else 'data/datasets'
os.makedirs(savepkl, exist_ok=True)

for env_name in ['antmaze-umaze', 'antmaze-medium', 'antmaze-large']:
	if env_name == 'antmaze-umaze':
		for dataset_type in ['', '-diverse', '']:
			name = f'{env_name}{dataset_type}-v0'
			download(name, args.is_awr)

	else :
		for dataset_type in ['diverse']:
			name = f'{env_name}-{dataset_type}-v0'
			download(name, args.is_awr)

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




    	 
				
             