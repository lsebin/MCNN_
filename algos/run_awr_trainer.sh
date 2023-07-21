task='antmaze-umaze-v0' #'halfcheetah-medium-replay-v2' #'antmaze-medium-play-v2' #'halfcheetah-medium-replay-v2'
AlgoType='awr'
mkdir algos/logs/${task}
mkdir algos/logs/${task}/${AlgoType}

SEED=1
Lipz=1
lamda=1

GPU=1

for lamda in 1 0.1
do
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/awr_trainer_new.py --task ${task} --Lipz ${Lipz} --lamda ${lamda}> algos/logs/${task}/mem_awr/Lipz${Lipz}_lamda${lamda}_seed${SEED}.log &
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/awr_trainer_og.py --task ${task} --Lipz ${Lipz} --lamda ${lamda}> algos/logs/${task}/awr/Lipz${Lipz}_lamda${lamda}_seed${SEED}.log &
done