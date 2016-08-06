#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

D="$(dirname $(readlink -f $0))"
NET_NAME=mv_lstm_vec_net
EXP_DETAIL=max_5_views_no_rnd_bg
OUT_PATH='/scratch/chrischoy/3DEverything/'$NET_NAME/$EXP_DETAIL

# Make the dir if it not there
mkdir -p $OUT_PATH
# LOG="$OUT_PATH/log.`date +'%Y-%m-%d_%H-%M-%S'`"
# exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

export THEANO_FLAGS="floatX=float32,device=gpu,assert_no_cpu_op='raise'"

python3 ./tools/analysis.py \
       --cfg ./experiments/cfgs/shapenet_1000.yaml \
       --cfg ./experiments/cfgs/no_random_background.yaml \
       --cfg ./experiments/cfgs/max_5_views.yaml \
       --cfg ./experiments/cfgs/local_shapenet.yaml \
       --out $OUT_PATH \
       --net $NET_NAME \
       --model $NET_NAME \
       ${*:1}
