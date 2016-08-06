#!/usr/bin/env python3

import _init_paths
import sys
import numpy as np
import argparse
import pprint

# Theano
import theano.sandbox.cuda
from lib.config import cfg, cfg_from_file, cfg_from_list
from lib.test_net import test_net

PY2 = sys.version_info[0] == 2
np.set_printoptions(precision=4)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a 3Deverything network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [gpu0]',
                        default='gpu0', type=str)
    parser.add_argument('--cfg', dest='cfg_files', action='append',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--net', dest='net_name',
                        help='name of the net',
                        default=None, type=str)
    parser.add_argument('--model', dest='model_name',
                        help='name of the network model',
                        default=None, type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset config file',
                        default=None, type=str)
    parser.add_argument('--exp', dest='exp',
                        help='name of the experiment',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--weights', dest='weights',
                        help='Initialize network from the weights file', default=None)
    parser.add_argument('--out', dest='out_path',
                        help='set output path', default=cfg.DIR.OUT_PATH)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print('Called with args:')
    print(args)

    if not args.gpu_id:
        theano.sandbox.cuda.use(cfg.CONST.DEVICE)
    else:
        theano.sandbox.cuda.use(args.gpu_id)

    for cfg_file in args.cfg_files:
        cfg_from_file(cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if args.net_name is not None:
        cfg.NET_NAME = args.net_name
    if args.model_name is not None:
        cfg.CONST.RECNET = args.model_name
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
    if args.dataset is not None:
        cfg.DATASET = args.dataset
    if args.exp is not None:
        cfg.TEST.EXP_NAME = args.exp
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path

    print('Using config:')
    pprint.pprint(cfg)

    test_net()

if __name__ == '__main__':
    main()
