mkdir algos/logs
mkdir algos/exp

percent=1.0
AlgoType=mopo_memdynamic_policy # bc OR td3bc
SEED=1


task='antmaze-umaze-v0' #'halfcheetah-medium-replay-v2' #'antmaze-medium-play-v2' #'halfcheetah-medium-replay-v2'
mkdir algos/logs/${task}
mkdir algos/logs/${task}/${AlgoType}

lamda=0.1
F=0.1
Lipz=1.0
rr=0.75
rollout=3

for F in 0.1 
 do
      for penalty in 0
      do
          GPU=0
          #CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_mnm_new.py --task ${task} --rollout-gamma ${gamma} --real-ratio ${rr} --rollout-length ${rollout} --penalty-coef ${penalty} --Lipz ${Lipz} --lamda ${lamda} --num_memories_frac ${F}> algos/logs/${task}/${AlgoType}/frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}_real-ratio${rr}_coef${penalty}_rollout${rollout}_gamma${gamma}_v2.log &
          #CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/run_mnm_new.py --task ${task} --real-ratio ${rr} --rollout-length ${rollout} --penalty-coef ${penalty} --Lipz ${Lipz} --lamda ${lamda} --num_memories_frac ${F}> algos/logs/${task}/${AlgoType}/frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}_real-ratio${rr}_coef${penalty}_rollout${rollout}_v2.log &
      done
 done




# hopper-medium-replay-v2 walker2d-medium-replay-v2 halfcheetah-medium-replay-v2 
# hopper-expert-v2 walker2d-expert-v2 halfcheetah-expert-v2 
# hopper-medium-v2 walker2d-medium-v2 halfcheetah-medium-v2 
