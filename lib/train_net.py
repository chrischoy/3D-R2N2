import os
import time
import numpy as np
from multiprocessing import Queue
from datetime import datetime

# Training related functions
from lib.solver import Solver
from lib.data_process import DataProcess
from lib.config import cfg
from lib.utils import Timer
from lib.data_io import category_model_id_pair, get_voc2012_imglist
import importlib


# Create worker and data queue for data processing
train_data_queue = Queue(cfg.QUEUE_SIZE)
test_data_queue = Queue(cfg.QUEUE_SIZE)

train_data_processes = []
test_data_processes = []


def max_or_nan(params):
    for param_idx, param in enumerate(params):
        # If there is nan, max will return nan
        nan_or_max_param = np.max(np.abs(param.val.get_value()))
        print('param %d : %f' % (param_idx, nan_or_max_param))
    return nan_or_max_param


def save_network(net, loss_all, param_save_dir, train_ind):
    save_path = os.path.join(param_save_dir, 'weights.%d' % (train_ind))
    net.save(save_path)

    # Make a symlink for weights.npy
    symlink_path = os.path.join(param_save_dir, 'weights.npy')
    if os.path.lexists(symlink_path):
        os.remove(symlink_path)

    os.symlink("%s.npy" % os.path.abspath(save_path), symlink_path)

    with open(os.path.join(param_save_dir, 'loss.%d.txt' % train_ind), 'w') as f:
        f.write('\n'.join([str(l) for l in loss_all]))


def kill_processes(queue, processes):
    print('Signal processes')
    for p in processes:
        p.shutdown()

    print('Empty queue')
    while not queue.empty():
        time.sleep(0.5)
        queue.get(False)

    print('kill processes')
    for p in processes:
        p.terminate()


def keyboard_interrupt_handle(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except(KeyboardInterrupt):
            print('Wait until the dataprocesses to end')
            kill_processes(train_data_queue, train_data_processes)
            kill_processes(test_data_queue, test_data_processes)

    return func_wrapper


@keyboard_interrupt_handle
def train_net():
    # Training
    loss_all = []

    training_timer = Timer()
    data_timer = Timer()

    param_save_dir = os.path.join(cfg.DIR.OUT_PATH)

    if not os.path.exists(param_save_dir):
        os.makedirs(param_save_dir)

    # set up training parameters
    lr = cfg.TRAIN.DEFAULT_LEARNING_RATE
    lr_steps = [int(k) for k in cfg.TRAIN.LEARNING_RATES.keys()]

    # set up model
    netlib = importlib.import_module("models.%s" % (cfg.CONST.RECNET))

    # print network definition
    print('Print network definition\n')
    with open('models/%s.py' % cfg.CONST.RECNET, 'r') as f:
        print(''.join(f.readlines()))

    net = netlib.RecNet()
    if net.is_x_tensor4 and cfg.CONST.N_VIEWS > 1:
        raise ValueError('Do not set the config.CONST.N_VIEWS > 1 when using \
                         single-view reconstruction network')

    solver = Solver(net)
    solver.compile_model(cfg.TRAIN.POLICY)

    start_iter = 0
    if cfg.TRAIN.RESUME_TRAIN:
        net_fn = os.path.join(cfg.CONST.WEIGHTS)
        net.load(net_fn)
        start_iter = cfg.TRAIN.INITIAL_ITERATION

    print('Set the learning rate to %f.' % lr)
    solver.set_lr(lr)

    # Start prefetching data processes
    train_category_model_pair = \
        category_model_id_pair(dataset_portion=cfg.TRAIN.DATASET_PORTION)
    for i in range(cfg.TRAIN.NUM_WORKER):
        bg_list = []
        if cfg.TRAIN.RANDOM_BACKGROUND:
            bg_list = get_voc2012_imglist()
        data_process = DataProcess(train_data_queue, train_category_model_pair, bg_list)
        data_process.start()
        train_data_processes.append(data_process)

    test_category_model_pair = \
        category_model_id_pair(dataset_portion=cfg.TEST.DATASET_PORTION)
    test_data_process = DataProcess(test_data_queue, test_category_model_pair, bg_list)
    test_data_process.start()
    test_data_processes.append(test_data_process)

    # Main training loop
    for train_ind in range(start_iter, cfg.TRAIN.NUM_ITERATION):
        data_timer.tic()
        batch_img, batch_voxel = train_data_queue.get()

        if net.is_x_tensor4:
            batch_img = batch_img[0]
        data_timer.toc()

        # Decrease learning rate at certain points
        if train_ind in lr_steps:
            solver.set_lr(np.float32(cfg.TRAIN.LEARNING_RATES[str(train_ind)]))  # edict only takes string for key
            print('Learing rate decreased to %f: ' % solver.lr.get_value())

        training_timer.tic()
        loss = solver.train_loss(batch_img, batch_voxel)
        training_timer.toc()

        loss_all.append(loss)

        if train_ind % cfg.TRAIN.PRINT_FREQ == 0:
            print('%s Iter: %d Loss: %f' % (datetime.now(), train_ind, loss))
            if data_timer.diff > 0.1:
                print('Warning: Prefetching blocking time: %s Increase the number of workers' % str(data_timer.diff))

        # Print test loss and params to check convergence every N iterations
        if train_ind % 2000 == 0:
            val_losses = []
            for i in range(cfg.TRAIN.NUM_VALIDATION_ITERATIONS):
                batch_img, batch_voxel = test_data_queue.get()
                prediction, val_loss, activations = solver.test_output(batch_img, batch_voxel)
                val_losses.append(val_loss)

            print('%s Test loss: %f' % (datetime.now(), np.mean(val_losses)))

            max_param = max_or_nan(net.params)
            if np.isnan(max_param):
                print('nan detected')
                break

        if train_ind % cfg.TRAIN.SAVE_FREQ == 0 and not train_ind == 0:
            save_network(net, loss_all, param_save_dir, train_ind)

        if loss > 2:
            print("Cost exceeds the threshold. Stop training")
            break

    save_network(net, loss_all, param_save_dir, train_ind)

    kill_processes(train_data_queue, train_data_processes)
    kill_processes(test_data_queue, test_data_processes)


def main():
    """ Test function"""
    cfg.DATASET = '/cvgl/group/ShapeNet/ShapeNetCore.v1/cat1000.json'
    cfg.CONST.RECNET = 'rec_net_2'
    cfg.TRAIN.DATASET_PORTION = [0, 0.8]
    train_net()

if __name__ == '__main__':
    main()
