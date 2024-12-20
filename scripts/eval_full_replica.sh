#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

for scene_num in 0 1 2 3 4 5 6 7
do
    for seed in 0 1 2
    do
        SCENE_NUM=${scene_num}
        export SCENE_NUM
        SEED=${seed}
        export SEED
        echo "Running scene number ${SCENE_NUM} with seed ${SEED}"
        python3 -u run.py configs/Replica/replica.py
    done
done