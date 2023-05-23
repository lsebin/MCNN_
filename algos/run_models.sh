mkdir algos/logs

TASK=halfcheetah-medium-v2
TSIZE=0.2
PHASE=train_dynamics

mkdir algos/logs/${TASK}_${TSIZE}_train

# -------- mopo ------------
GPU=0
frac=0.025
# CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/mopo_trainer.py --algo-name mopo --phase ${PHASE} --task ${TASK} --train-size ${TSIZE} --use_tqdm 0 --num_memories_frac ${frac} > algos/logs/${TASK}_${TSIZE}_train/${PHASE}_vanilla_ensemble.log &

# # -------- deterministic memensemble_mopo --------
# frac=0.05
# Lipz=2.5
# lamda=10.0
# GPU=1
# CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/mopo_deterministic_trainer.py --algo-name memensemble_mopo --phase ${PHASE} --task ${TASK} --train-size ${TSIZE} --use_tqdm 0 --num_memories_frac ${frac} --Lipz ${Lipz} --lamda ${lamda} > algos/logs/${TASK}_${TSIZE}_train/${PHASE}_deterministicmemensemble_frac${frac}_Lipz${Lipz}_lamda${lamda}.log &

for frac in 0.025 # options: 0.025 0.05 0.1 
do
    # -------- memensemble_mopo --------
    for Lipz in 2.5 5.0 10.0 15.0
    do 
        for lamda in 0.1 1.0
        do
            GPU=0
            # CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/mopo_trainer.py --algo-name memensemble_mopo --phase ${PHASE} --task ${TASK} --train-size ${TSIZE} --use_tqdm 0 --num_memories_frac ${frac} --Lipz ${Lipz} --lamda ${lamda} > algos/logs/${TASK}_${TSIZE}_train/${PHASE}_memensemble_frac${frac}_Lipz${Lipz}_lamda${lamda}.log &
        done
    done

    for Lipz in 2.5 15.0
    do 
        for lamda in 10.0
        do
            GPU=1
            # CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/mopo_trainer.py --algo-name memensemble_mopo --phase ${PHASE} --task ${TASK} --train-size ${TSIZE} --use_tqdm 0 --num_memories_frac ${frac} --Lipz ${Lipz} --lamda ${lamda} > algos/logs/${TASK}_${TSIZE}_train/${PHASE}_memensemble_frac${frac}_Lipz${Lipz}_lamda${lamda}.log &
        done
    done

    # -------- mem_mopo ------------
    for Lipz in 2.5 5.0
    do 
        for lamda in 0.1 1.0 10.0
        do
            GPU=1
            CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/mopo_trainer.py --algo-name mem_mopo --phase ${PHASE} --task ${TASK} --train-size ${TSIZE} --use_tqdm 0 --num_memories_frac ${frac} --Lipz ${Lipz} --lamda ${lamda} > algos/logs/${TASK}_${TSIZE}_train/${PHASE}_mem_frac${frac}_Lipz${Lipz}_lamda${lamda}.log &
        done
    done

    for Lipz in 10.0 15.0
    do 
        for lamda in 0.1 1.0 10.0
        do
            GPU=0
x        done
    done
done 