mkdir algos/logs

TASK=halfcheetah-medium-v2
TSIZE=0.2
PHASE=test_mppi

mkdir algos/logs/${TASK}_${TSIZE}_train


# -------- memensemble_mopo --------
frac=0.05
Lipz=2.5
lamda=10.0
for P in 0.5
do
    GPU=0
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/mopo_trainer.py --algo-name memensemble_mopo --phase ${PHASE} --task ${TASK} --train-size ${TSIZE} --use_tqdm 0 --num_memories_frac ${frac} --Lipz ${Lipz} --lamda ${lamda} --penalty-coef ${P} > algos/logs/${TASK}_${TSIZE}_train/${PHASE}_memensemble_penalty${P}_frac${frac}_Lipz${Lipz}_lamda${lamda}.log &
done

# -------- mem_mopo ------------
frac=0.1
Lipz=15.0
lamda=1.0
for P in 0.5 1.0 2.0 4.0
do
    GPU=1
    # CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/mopo_trainer.py --algo-name mem_mopo --phase ${PHASE} --task ${TASK} --train-size ${TSIZE} --use_tqdm 0 --num_memories_frac ${frac} --Lipz ${Lipz} --lamda ${lamda} --penalty-coef ${P} > algos/logs/${TASK}_${TSIZE}_train/${PHASE}_mem_penalty${P}_frac${frac}_Lipz${Lipz}_lamda${lamda}.log &
done 

