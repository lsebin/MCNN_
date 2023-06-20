#mkdir algos/logs
#mkdir algos/exp


percent=1.0
AlgoType=combo_memdynamic_policy # bc OR td3bc
SEED=1

# hammer-human-v1 pen-human-v1 relocate-human-v1 door-human-v1
# hammer-expert-v1 pen-expert-v1 relocate-expert-v1 door-expert-v1
# hammer-cloned-v1 pen-cloned-v1 relocate-cloned-v1 door-cloned-v1
# carla-lane-v0 
# carla-town-v0

task='halfcheetah-medium-replay-v2'
mkdir algos/logs/${task}/${AlgoType}


# for TASK in halfcheetah walker2d hopper
# do
#     GPU=1
#     CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_combo.py --task ${TASK}-medium-replay-v2 > algos/logs/mem_${AlgoType}_${TASK}-medium-replay-v2_frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}.log &
# done

F=0.1

rollout=30
rr=0.5
Lipz=1.0
lamda=1.0

# for lamda in 0.1
# do
#      for Lipz in 1.0
#      do
#          GPU=0
#          CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_combo.py --task ${task} --real-ratio ${rr} --rollout-length ${rollout} --Lipz ${Lipz} --lamda ${lamda} > algos/logs/${task}/${AlgoType}/frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}_real-ratio${rr}_rollout${rollout}_norm.log &
#      done
# done

# Lipz=1.0
# lamda=1.0

# for rollout in 10 20 30 40
# do
#     for rr in 0.75
#     do
#         GPU=0
#         CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_combo.py --task ${task} --real-ratio ${rr} --rollout-length ${rollout} --use-tqdm 0 > algos/logs/${task}/${AlgoType}/frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}_real-ratio${rr}_rollout${rollout}.log &
#     done
# done

  for rollout in 10
  do
       for rr in 0.25
       do
           GPU=1
           CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_combo.py --task ${task} --real-ratio ${rr} --rollout-length ${rollout} --Lipz ${Lipz} --lamda ${lamda} > algos/logs/${task}/${AlgoType}/frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}_real-ratio${rr}_rollout${rollout}_norm_v2.log &
       done
   done


# F=0.1
# for coef in 0.01 0.025 0.05 0.075 0.1
# do
#     GPU=1
#     CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/combo_trainer.py  --penalty-coef ${coef} --use-tqdm 0 > algos/logs/mem_${AlgoType}_${task}_frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}_penalty_coef${coef}.log &
# done 



# F=0.05
# for task in ${TASKS}
# do
#     for Lipz in 1.0
#     do 
#         for lamda in 1.0
#         do 
#             GPU=0
#             CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/combo_trainer.py --seed ${SEED} --chosen-percentage ${percent} --algo-name mem_${AlgoType} --task ${task} --num_memories_frac ${F} --Lipz ${Lipz} --lamda ${lamda} --use-tqdm 0 > algos/logs_td3bc/mem_${AlgoType}_${task}_frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}.log &
#         done
#     done
# done 


# F=0.025
# for task in ${TASKS}
# do
#     for Lipz in 1.0
#     do 
#         for lamda in 1.0
#         do 
#             GPU=0
#             CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/combo_trainer.py --seed ${SEED} --chosen-percentage ${percent} --algo-name mem_${AlgoType} --task ${task} --num_memories_frac ${F} --Lipz ${Lipz} --lamda ${lamda} --use-tqdm 0 > algos/logs_td3bc/mem_${AlgoType}_${task}_frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}.log &
#         done
#     done
# done 


# hopper-medium-replay-v2 walker2d-medium-replay-v2 halfcheetah-medium-replay-v2 
# hopper-expert-v2 walker2d-expert-v2 halfcheetah-expert-v2 
# hopper-medium-v2 walker2d-medium-v2 halfcheetah-medium-v2 
