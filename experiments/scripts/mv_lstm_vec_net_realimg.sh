#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

D="$(dirname $(readlink -f $0))"
NET_NAME=mv_lstm_vec_net
OUT_PATH='./output/'$NET_NAME
LOG="$OUT_PATH/log.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

export THEANO_FLAGS="float32,device=gpu,assert_no_cpu_op='raise'"

python3 ./tools/train.py \
       --cfg ./experiments/cfgs/random_crop.yaml \
       --cfg ./experiments/cfgs/mv_lstm_vec_net.yaml \
       --cfg ./experiments/cfgs/finetune_realimg.yaml \
       --net $NET_NAME \
       ${*:1}
