#!/bin/bash

docker run -it -d --rm --gpus=all \
    --shm-size=64g \
    -p 8888:8888  \
    --name pytorch_cv_project \
    -v /data/datasets:/datasets:rw \
    -v /home/melekhin/raai-cv-pose-estimation/configs:/configs:ro \
    -v /home/melekhin/raai-cv-pose-estimation/baselines:/baselines:rw \
    -v /home/melekhin/raai-cv-pose-estimation/notebooks:/notebooks:rw \
    -v /home/melekhin/raai-cv-pose-estimation/src:/src:rw \
    -v /home/melekhin/raai-cv-pose-estimation/exp:/exp:rw \
    -v /data/datasets/COCO2017:/baselines/AlphaPose/data/coco:ro \
    -v /home/melekhin/datasets/MPII:/baselines/AlphaPose/data/mpii:ro \
    -v /data/datasets/COCO2017:/data/coco:ro \
    -v /home/melekhin/datasets/MPII:/data/mpii:ro \
    cv_pose_estimation
