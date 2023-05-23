mkdir algos/logs_td3bc
mkdir algos/exp_td3bc

percent=1.0
AlgoType=bc # bc OR td3bc
SEED=4

# hammer-human-v1 pen-human-v1 relocate-human-v1 door-human-v1
# hammer-expert-v1 pen-expert-v1 relocate-expert-v1 door-expert-v1
# hammer-cloned-v1 pen-cloned-v1 relocate-cloned-v1 door-cloned-v1
# carla-lane-v0 
# carla-town-v0

TASKS='hammer-human-v1 pen-human-v1 relocate-human-v1 door-human-v1 hammer-expert-v1 pen-expert-v1 relocate-expert-v1 door-expert-v1 hammer-cloned-v1 pen-cloned-v1 relocate-cloned-v1 door-cloned-v1 carla-lane-v0 carla-town-v0'

F=0.1
for task in ${TASKS}
do
    GPU=0
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/td3bc_trainer.py --seed ${SEED} --chosen-percentage ${percent} --algo-name ${AlgoType} --task ${task} --use-tqdm 0 > algos/logs_td3bc/${AlgoType}_${task}_seed${SEED}.log &

    for Lipz in 1.0
    do 
        for lamda in 1.0
        do 
            GPU=1
            CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/td3bc_trainer.py --seed ${SEED} --chosen-percentage ${percent} --algo-name mem_${AlgoType} --task ${task} --num_memories_frac ${F} --Lipz ${Lipz} --lamda ${lamda} --use-tqdm 0 > algos/logs_td3bc/mem_${AlgoType}_${task}_frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}.log &
        done
    done
done 


F=0.05
for task in ${TASKS}
do
    for Lipz in 1.0
    do 
        for lamda in 1.0
        do 
            GPU=0
            CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/td3bc_trainer.py --seed ${SEED} --chosen-percentage ${percent} --algo-name mem_${AlgoType} --task ${task} --num_memories_frac ${F} --Lipz ${Lipz} --lamda ${lamda} --use-tqdm 0 > algos/logs_td3bc/mem_${AlgoType}_${task}_frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}.log &
        done
    done
done 


F=0.025
for task in ${TASKS}
do
    for Lipz in 1.0
    do 
        for lamda in 1.0
        do 
            GPU=0
            CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/td3bc_trainer.py --seed ${SEED} --chosen-percentage ${percent} --algo-name mem_${AlgoType} --task ${task} --num_memories_frac ${F} --Lipz ${Lipz} --lamda ${lamda} --use-tqdm 0 > algos/logs_td3bc/mem_${AlgoType}_${task}_frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}.log &
        done
    done
done 


# hopper-medium-replay-v2 walker2d-medium-replay-v2 halfcheetah-medium-replay-v2 
# hopper-expert-v2 walker2d-expert-v2 halfcheetah-expert-v2 
# hopper-medium-v2 walker2d-medium-v2 halfcheetah-medium-v2 
