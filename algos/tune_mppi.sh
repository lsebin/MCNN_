mkdir algos/logs

TASK=halfcheetah-medium-v2
TSIZE=0.2
PHASE=tune_mppi

mkdir algos/logs/${TASK}_${TSIZE}_train

# -------- mopo ------------
GPU=0
frac=0.025
# CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/mopo_trainer.py --algo-name mopo --phase ${PHASE} --task ${TASK} --train-size ${TSIZE} --use_tqdm 0 --num_memories_frac ${frac} > algos/logs/${TASK}_${TSIZE}_train/${PHASE}_vanilla_ensemble.log &


# -------- mem_mopo ----------
frac=0.1
Lipz=15.0
lamda=1.0
for P in 20.0
do
    GPU=1
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/mopo_trainer.py --algo-name mem_mopo --phase ${PHASE} --task ${TASK} --train-size ${TSIZE} --use_tqdm 0 --num_memories_frac ${frac} --Lipz ${Lipz} --lamda ${lamda} --penalty-coef ${P} > algos/logs/${TASK}_${TSIZE}_train/${PHASE}_mem_penalty${P}_frac${frac}_Lipz${Lipz}_lamda${lamda}.log &
done 

for P in 40.0 64.0
do
    GPU=0
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/mopo_trainer.py --algo-name mem_mopo --phase ${PHASE} --task ${TASK} --train-size ${TSIZE} --use_tqdm 0 --num_memories_frac ${frac} --Lipz ${Lipz} --lamda ${lamda} --penalty-coef ${P} > algos/logs/${TASK}_${TSIZE}_train/${PHASE}_mem_penalty${P}_frac${frac}_Lipz${Lipz}_lamda${lamda}.log &
done 

