#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

for scene_num in 0 1 2 3 4 5
do
    for seed in 0 1 2
    do
        SCENE_NUM=${scene_num}
        export SCENE_NUM
        SEED=${seed}
        export SEED
        echo "Running scene number ${SCENE_NUM} with seed ${SEED}"
        python3 -u run.py configs/Scannet/scannet.py
    done
done