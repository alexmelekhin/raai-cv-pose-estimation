docker run -it --rm --gpus=all \
    -p 8888:8888  \
    --name pytorch_cv_project \
    -v /data/datasets:/datasets:rw \
    -v /home/melekhin/raai-cv-pose-estimation/baselines:/baselines:rw \
    pytorch-cuda
