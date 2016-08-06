#!/usr/bin/env python3

import _init_paths
import sys
import numpy as np

import multiprocessing
import logging
import argparse
import pprint

import theano.sandbox.cuda as cuda
from lib.config import cfg, cfg_from_file, cfg_from_list
from lib.train_net import train_net

PY2 = sys.version_info[0] == 2

np.set_printoptions(precision=4)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a 3Deverything network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [gpu0]',
                        default='gpu0', type=str)
    parser.add_argument('--cfg', dest='cfg_files', action='append',
                        help='optional config files',
                        default=None, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset config file',
                        default=None, type=str)
    parser.add_argument('--model', dest='model_name',
                        help='name of the network model',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--out', dest='out_path',
                        help='set output path', default=cfg.DIR.OUT_PATH)
    parser.add_argument('--weights', dest='weights',
                        help='Initialize network from the weights file', default=None)
    parser.add_argument('--iter', dest='init_iter',
                        help='Start from the specified iteration', default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print('Called with args:')
    print(args)

    if not args.gpu_id:
        cuda.use(cfg.CONST.DEVICE)
    else:
        cuda.use(args.gpu_id)

    if args.cfg_files is not None:
        for cfg_file in args.cfg_files:
            cfg_from_file(cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if args.model_name is not None:
        cfg.CONST.RECNET = args.model_name
    if not args.randomize:
        np.random.seed(cfg.CONST.RNG_SEED)
    if args.dataset is not None:
        cfg.DATASET = args.dataset
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.weights is not None:
        cfg.TRAIN.RESUME_TRAIN = True
        cfg.CONST.WEIGHTS = args.weights
        cfg.TRAIN.INITIAL_ITERATION = int(args.init_iter)

    print('Using config:')
    pprint.pprint(cfg)

    train_net()


if __name__ == '__main__':
    multiprocessing.log_to_stderr()
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    main()
