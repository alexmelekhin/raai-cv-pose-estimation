#!/bin/bash

docker run -it -d --rm --gpus=all \
    --shm-size=64g \
    -p 8888:8888  \
    --name pytorch_cv_project \
    -v /data/datasets:/datasets:rw \
    -v /home/melekhin/raai-cv-pose-estimation/custom_alphapose_configs:/baselines/AlphaPose/configs/custom:ro \
    -v /home/melekhin/raai-cv-pose-estimation/baselines:/baselines:rw \
    -v /data/datasets/COCO2017:/baselines/AlphaPose/data/coco:ro \
    -v /data/datasets/MPII:/baselines/AlphaPose/data/mpii:ro \
    cv_pose_estimation
