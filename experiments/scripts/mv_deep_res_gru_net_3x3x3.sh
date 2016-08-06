#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

D="$(dirname $(readlink -f $0))"
NET_NAME=mv_deep_res_gru_net_3x3x3
EXP_DETAIL=max_5_views_no_rnd_bg
OUT_PATH='./output/'$NET_NAME/$EXP_DETAIL
LOG="$OUT_PATH/log.`date +'%Y-%m-%d_%H-%M-%S'`"

# Make the dir if it not there
mkdir -p $OUT_PATH
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

export THEANO_FLAGS="floatX=float32,device=gpu,assert_no_cpu_op='raise'"

python main.py \
      --batch-size 24 \
      --cfg ./experiments/cfgs/shapenet_1000.yaml \
      --cfg ./experiments/cfgs/random_crop.yaml \
      --cfg ./experiments/cfgs/no_random_background.yaml \
      --cfg ./experiments/cfgs/max_5_views.yaml \
      --cfg ./experiments/cfgs/local_shapenet.yaml \
      --out $OUT_PATH \
      --model $NET_NAME \
      ${*:1}

python main.py \
      --test \
      --batch-size 24 \
      --cfg ./experiments/cfgs/shapenet_1000.yaml \
      --cfg ./experiments/cfgs/no_random_background.yaml \
      --cfg ./experiments/cfgs/max_5_views.yaml \
      --cfg ./experiments/cfgs/local_shapenet.yaml \
      --weights $OUT_PATH/weights.npy \
      --model $NET_NAME \
      ${*:1}
