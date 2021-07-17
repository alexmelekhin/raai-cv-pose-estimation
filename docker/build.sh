#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

if command -v nvidia-smi &> /dev/null
then
    echo "Building for ${orange}nvidia${reset_color} hardware"
    DOCKERFILE=Dockerfile.nvidia
    DEVICE=cuda
else
    echo "Building for ${orange}intel${reset_color} hardware: nvidia driver not found"
    DOCKERFILE=Dockerfile.intel
    DEVICE=cpu
fi

docker build . \
    -f $DOCKERFILE \
    -t cv_pose_estimation:$DEVICE-latest
