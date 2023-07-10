mkdir mems_obs/logs_re

 for ENV in hopper #walker2d halfcheetah
 do
     for TYPE in medium-replay # medium expert random medium-replay medium-expert
     do
         mkdir mems_obs/logs_re/${ENV}-${TYPE}-v2
         for F in 0.2 #0.3 0.4  #0.01 0.025 0.05 0.1 0.25 0.3
         do
            CUDA_VISIBLE_DEVICES=0 nohup python -u mems_obs/update_data.py --name ${ENV}-${TYPE}-v2 --num_memories_frac ${F} > mems_obs/logs/${ENV}-${TYPE}-v2/update_data_${F}_frac_percentbc.log &
         done 
     done
 done


# for ENV in hammer pen relocate door
# do
#     for TYPE in expert cloned human
#     do
#         mkdir mems_obs/logs/${ENV}-${TYPE}-v1
#         for F in 0.025 0.05 0.1
#         do
#             CUDA_VISIBLE_DEVICES=1 nohup python -u mems_obs/update_data.py --name ${ENV}-${TYPE}-v1 --num_memories_frac ${F} > mems_obs/logs/${ENV}-${TYPE}-v1/update_data_${F}_frac_percentbc.log &
#         done 
#     done
# done


# for TASK in carla-lane-v0 carla-town-v0 
# do
#     mkdir mems_obs/logs/${TASK}
#     for F in 0.01 0.025 0.05 0.1
#     do
#         CUDA_VISIBLE_DEVICES=0 nohup python -u mems_obs/update_data.py --name ${TASK} --num_memories_frac ${F} > mems_obs/logs/${TASK}/update_data_${F}_frac_percentbc.log &
#     done 
# done


# for TYPE in diverse-v2 v2
#     mkdir mems_obs/logs/antmaze-umaze-${TYPE}
#     for F in 0.01 0.025 0.05 0.1
#     do
#         nohup python -u mems_obs/update_data.py --name antmaze-umaze-${TYPE} --num_memories_frac ${F} > mems_obs/logs/antmaze-umaze-${TYPE}/update_data_${F}_frac_percentbc.log &
#     done 
# done


# for ENV in antmaze-large antmaze-medium
# do
#     for TYPE in diverse play
#         mkdir mems_obs/logs/${ENV}-${TYPE}-v2
#         for F in 0.01 0.025 0.05 0.1
#         do
#             nohup python -u mems_obs/update_data.py --name ${ENV}-${TYPE}-v2 --num_memories_frac ${F} > mems_obs/logs/${ENV}-${TYPE}-v2/update_data_${F}_frac_percentbc.log &
#         done 
#     done
# done