#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

if command -v nvidia-smi &> /dev/null
then
    echo "Running on ${orange}nvidia${reset_color} hardware"
    ARGS="--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all"
    DEVICE=cuda
else
    echo "Running on ${orange}intel${reset_color} hardware: nvidia driver not found"
    ARGS="--device=/dev/dri:/dev/dri"
    DEVICE=cpu

xhost +
docker run -itd --rm \
    $ARGS \
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
    cv_pose_estimation:$DEVICE-latest
xhost -

# -v /data/datasets/COCO2017:/baselines/AlphaPose/data/coco:ro \
# -v /home/melekhin/datasets/MPII:/baselines/AlphaPose/data/mpii:ro \
# -v /data/datasets/COCO2017:/data/coco:ro \
# -v /home/melekhin/datasets/MPII:/data/mpii:ro \