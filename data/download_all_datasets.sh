mkdir data/log

name='antmaze-medium-diverse'
awr=true

for name in 'antmaze-medium-diverse' 'antmaze-large-diverse'
do
    nohup python -u data/download_d4rl_datasets.py --name ${name}-v0 --is_awr ${awr} > data/log/${name}-v0_is_awr${awr}.log &
done