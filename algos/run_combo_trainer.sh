#mkdir algos/logs
#mkdir algos/exp


percent=1.0
AlgoType=mopo_memdynamic_policy # bc OR td3bc
SEED=1

# hammer-human-v1 pen-human-v1 relocate-human-v1 door-human-v1
# hammer-expert-v1 pen-expert-v1 relocate-expert-v1 door-expert-v1
# hammer-cloned-v1 pen-cloned-v1 relocate-cloned-v1 door-cloned-v1
# carla-lane-v0 
# carla-town-v0

task='antmaze-umaze-v0' #'halfcheetah-medium-replay-v2' #'antmaze-medium-play-v2' #'halfcheetah-medium-replay-v2'
mkdir algos/logs/${task}
mkdir algos/logs/${task}/${AlgoType}

lamda=0.1
F=0.1
Lipz=1.0
rr=0.75
rollout=3

# AlgoType=combo_original
# AlgoType=cql_original
# mkdir algos/logs/${task}/${AlgoType}
GPU=1
CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_combo.py --task ${task} --real-ratio ${rr} --rollout-length ${rollout} --Lipz ${Lipz} --lamda ${lamda} --num_memories_frac ${F}> algos/logs/${task}/${AlgoType}/frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}_real-ratio${rr}_coef${penalty}_rollout${rollout}_gamma${gamma}_v2.log &
# CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_cql_original.py --task ${task} --seed ${SEED}> algos/logs/${task}/${AlgoType}/seed${SEED}.log &
#CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_combo_original.py --task ${task} --Lipz ${Lipz} --lamda ${lamda} --num_memories_frac ${F} --seed ${SEED}> algos/logs/${task}/${AlgoType}/frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}.log &

# AlgoType=mopo_original
# mkdir algos/logs/${task}/${AlgoType}
GPU=0

#CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_mopo_original.py --task ${task} --Lipz ${Lipz} --lamda ${lamda} --num_memories_frac ${F} --seed ${SEED}> algos/logs/${task}/${AlgoType}/frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}.log &
# CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_mopo_original.py --task ${task} --seed ${SEED}> algos/logs/${task}/${AlgoType}/seed${SEED}_v2.log &

# for TASK in halfcheetah walker2d hopper
# do
#     GPU=1
#     CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_combo.py --task ${TASK}-medium-replay-v2 > algos/logs/mem_${AlgoType}_${TASK}-medium-replay-v2_frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}.log &
# done

F=0.1

# rollout=5
rr=0.75
Lipz=1.0
# lamda=1.0
# penalty=0

  # lamda=1.0
  # for F in 0.25
  # do
  #     for rollout in 3
  #     do
  #         GPU=1
  #         CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_combo.py --task ${task} --real-ratio ${rr} --rollout-length ${rollout} --Lipz ${Lipz} --lamda ${lamda} --num_memories_frac ${F} > algos/logs/${task}/${AlgoType}/frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}_real-ratio${rr}_rollout${rollout}_norm_v2.log &
  #     done
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

lamda=1.0
rollout=3
rr=0.75
F=0.1

 for F in 0.1 # 0.4 #0.3 0.2 #0.25
# #for penalty in 0.3 0.4 0.5 #0.15 0.125 0.01 0.1 0.025 0.075 0.05
 do
      for penalty in 0
      do
          GPU=0
          #CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_combo.py --task ${task} --rollout-gamma ${gamma} --real-ratio ${rr} --rollout-length ${rollout} --penalty-coef ${penalty} --Lipz ${Lipz} --lamda ${lamda} --num_memories_frac ${F}> algos/logs/${task}/${AlgoType}/frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}_real-ratio${rr}_coef${penalty}_rollout${rollout}_gamma${gamma}_v2.log &
          #CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_combo.py --task ${task} --real-ratio ${rr} --rollout-length ${rollout} --penalty-coef ${penalty} --Lipz ${Lipz} --lamda ${lamda} --num_memories_frac ${F}> algos/logs/${task}/${AlgoType}/frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}_real-ratio${rr}_coef${penalty}_rollout${rollout}_v2.log &
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
