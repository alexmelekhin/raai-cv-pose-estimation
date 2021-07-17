#!/bin/bash

xhost +
docker run -it -d --rm --gpus=all \
    --shm-size=64g \
    -p 8888:8888  \
    --net host \
    --ipc host \
    --privileged \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --name pytorch_cv_project \
    -v /mnt/workspace/melekhin/raai-cv-pose-estimation/ros_pkgs/pose_estimator:/catkin_ws/src/pose_estimator:rw \
    -v /mnt/workspace/melekhin/raai-cv-pose-estimation/configs:/catkin_ws/configs:ro \
    -v /mnt/workspace/melekhin/raai-cv-pose-estimation/weights:/catkin_ws/weights:ro \
    -v /mnt/workspace/datasets/:/datasets:rw \
    -v /mnt/workspace/melekhin/raai-cv-pose-estimation/configs:/configs:ro \
    -v /mnt/workspace/melekhin/raai-cv-pose-estimation/baselines:/baselines:rw \
    -v /mnt/workspace/melekhin/raai-cv-pose-estimation/notebooks:/notebooks:rw \
    -v /mnt/workspace/melekhin/raai-cv-pose-estimation/src:/src:rw \
    -v /mnt/workspace/melekhin/raai-cv-pose-estimation/exp:/exp:rw \
    cv_pose_estimation
xhost -

# -v /data/datasets/COCO2017:/baselines/AlphaPose/data/coco:ro \
# -v /home/melekhin/datasets/MPII:/baselines/AlphaPose/data/mpii:ro \
# -v /data/datasets/COCO2017:/data/coco:ro \
# -v /home/melekhin/datasets/MPII:/data/mpii:ro \