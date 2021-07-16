set -x

CONFIG=$1
EXPID=${2:-"alphapose"}

CUDA_VISIBLE_DEVICES=4 python3 ./src/train.py \
    --exp-id ${EXPID} \
    --cfg ${CONFIG} \
    --nThreads 96 \
    --snapshot 5