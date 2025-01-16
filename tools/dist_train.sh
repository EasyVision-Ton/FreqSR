#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
MASTER_PORT=${MASTER_PORT:-29509}

PYTHONPATH="$(dirname $0)/..":PYTHONPATH \
torchrun \
--nproc_per_node $GPUS \
--master_port $MASTER_PORT \
$(dirname $0)/main_train_psnr.py \
--opt $CONFIG \
--launcher pytorch ${@:3}


# python -m torch.distributed.launch \
# --nproc_per_node $GPUS \
# --master_port $MASTER_PORT \
# $(dirname $0)/main_train_psnr.py \
# --opt $CONFIG \
# --launcher pytorch ${@:3}
