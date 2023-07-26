task='antmaze-umaze-v0' #'halfcheetah-medium-replay-v2' #'antmaze-medium-play-v2' #'halfcheetah-medium-replay-v2'
AlgoType='awr'
mkdir algos/logs/${task}
mkdir algos/logs/${task}/${AlgoType}

SEED=1
Lipz=1
lamda=1.0
f=0.1
bs=512

GPU=1

for lamda in 0.1 1.0
do
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/awr_trainer_new.py --task ${task} --Lipz ${Lipz} --lamda ${lamda} --batch-size ${bs}> algos/logs/${task}/mem_awr/frac${f}_Lipz${Lipz}_lamda${lamda}_batch${bs}_seed${SEED}.log &
done

CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/awr_trainer_og.py --task ${task} --Lipz ${Lipz} --lamda ${lamda} --batch-size ${bs}> algos/logs/${task}/awr/frac${f}_Lipz${Lipz}_lamda${lamda}_batch${bs}_seed${SEED}.log &