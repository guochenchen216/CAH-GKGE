#!/bin/sh

dataset='10_m' # '', 'WN18RR', or 'nell'
kge='ComplEx' # 'TransE', 'DistMult', 'ComplEx', or 'ConvE'
CUDA_VISIBLE_DEVICES=6 nohup python main.py --data_name ${dataset} --name ${dataset,,}_${kge,,} \
        --step meta_train --kge ${kge} > result/train_ComplEx_10_m.out 2>&1 &