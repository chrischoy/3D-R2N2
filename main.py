#!/usr/bin/env python
import sys
if (sys.version_info < (3, 0)):
    raise Exception("Please follow the installation instruction on 'https://github.com/chrischoy/3D-R2N2'")

import numpy as np
import argparse
import pprint
import logging
import multiprocessing as mp

# Theano
import theano.sandbox.cuda
from lib.config import cfg, cfg_from_file, cfg_from_list
from lib.test_net import test_net
from lib.train_net import train_net


def parse_args():
    parser = argparse.ArgumentParser(description='Main 3Deverything train/test file')
    parser.add_argument(
        '--gpu',
        dest='gpu_id',
        help='GPU device id to use [gpu0]',
        default=cfg.CONST.DEVICE,
        type=str)
    parser.add_argument(
        '--cfg',
        dest='cfg_files',
        action='append',
        help='optional config file',
        default=None,
        type=str)
    parser.add_argument(
        '--rand', dest='randomize', help='randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument(
        '--test', dest='test', help='randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--net', dest='net_name', help='name of the net', default=None, type=str)
    parser.add_argument(
        '--model', dest='model_name', help='name of the network model', default=None, type=str)
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        help='name of the net',
        default=cfg.CONST.BATCH_SIZE,
        type=int)
    parser.add_argument(
        '--iter',
        dest='iter',
        help='number of iterations',
        default=cfg.TRAIN.NUM_ITERATION,
        type=int)
    parser.add_argument(
        '--dataset', dest='dataset', help='dataset config file', default=None, type=str)
    parser.add_argument(
        '--set', dest='set_cfgs', help='set config keys', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--exp', dest='exp', help='name of the experiment', default=None, type=str)
    parser.add_argument(
        '--weights', dest='weights', help='Initialize network from the weights file', default=None)
    parser.add_argument('--out', dest='out_path', help='set output path', default=cfg.DIR.OUT_PATH)
    parser.add_argument(
        '--init-iter',
        dest='init_iter',
        help='Start from the specified iteration',
        default=cfg.TRAIN.INITIAL_ITERATION)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print('Called with args:')
    print(args)

    # Set main gpu
    theano.sandbox.cuda.use(args.gpu_id)

    if args.cfg_files is not None:
        for cfg_file in args.cfg_files:
            cfg_from_file(cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if not args.randomize:
        np.random.seed(cfg.CONST.RNG_SEED)

    if args.batch_size is not None:
        cfg_from_list(['CONST.BATCH_SIZE', args.batch_size])
    if args.iter is not None:
        cfg_from_list(['TRAIN.NUM_ITERATION', args.iter])
    if args.net_name is not None:
        cfg_from_list(['NET_NAME', args.net_name])
    if args.model_name is not None:
        cfg_from_list(['CONST.NETWORK_CLASS', args.model_name])
    if args.dataset is not None:
        cfg_from_list(['DATASET', args.dataset])
    if args.exp is not None:
        cfg_from_list(['TEST.EXP_NAME', args.exp])
    if args.out_path is not None:
        cfg_from_list(['DIR.OUT_PATH', args.out_path])
    if args.weights is not None:
        cfg_from_list(['CONST.WEIGHTS', args.weights, 'TRAIN.RESUME_TRAIN', True,
                       'TRAIN.INITIAL_ITERATION', int(args.init_iter)])

    print('Using config:')
    pprint.pprint(cfg)

    if not args.test:
        train_net()
    else:
        test_net()


if __name__ == '__main__':
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)
    main()
