#!/bin/bash

time=$(date "+%Y%m%d%H%M%S")
rm -rf logs/$time
mkdir logs/$time/

python src/train.py     --dataset L4E \
                        --dataset_path /home/julian/data/lane_batch \
                        --train_list sample_gt \
                        --val_list val_gt \
                        --gpus 4 5 6 \
                        --save_path logs/$time \
2>&1|tee logs/$time/train_erfnet_L4E_$time.log

end_time=$(date "+%Y%m%d%H%M%S")
if (($end_time-$time < 3600))
then
    rm -rf logs/$time
fi