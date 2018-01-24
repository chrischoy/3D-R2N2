import os
import numpy as np
import scipy.io as sio
import inspect
import sklearn.metrics
from multiprocessing import Queue

# Theano & network
from models import load_model
from lib.config import cfg
from lib.solver import Solver
from lib.data_io import category_model_id_pair
from lib.data_process import make_data_processes, get_while_running

from lib.voxel import evaluate_voxel_prediction


def test_net():
    ''' Evaluate the network '''
    # Make result directory and the result file.
    result_dir = os.path.join(cfg.DIR.OUT_PATH, cfg.TEST.EXP_NAME)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_fn = os.path.join(result_dir, 'result.mat')

    print("Exp file will be written to: " + result_fn)

    # Make a network and load weights
    NetworkClass = load_model(cfg.CONST.NETWORK_CLASS)
    print('Network definition: \n')
    print(inspect.getsource(NetworkClass.network_definition))
    net = NetworkClass(compute_grad=False)
    net.load(cfg.CONST.WEIGHTS)
    solver = Solver(net)

    # set constants
    batch_size = cfg.CONST.BATCH_SIZE

    # set up testing data process. We make only one prefetching process. The
    # process will return one batch at a time.
    queue = Queue(cfg.QUEUE_SIZE)
    data_pair = category_model_id_pair(dataset_portion=cfg.TEST.DATASET_PORTION)
    processes = make_data_processes(queue, data_pair, 1, repeat=False, train=False)
    num_data = len(processes[0].data_paths)
    num_batch = int(num_data / batch_size)

    # prepare result container
    results = {'cost': np.zeros(num_batch),
               'mAP': np.zeros((num_batch, batch_size))}
    # Save results for various thresholds
    for thresh in cfg.TEST.VOXEL_THRESH:
        results[str(thresh)] = np.zeros((num_batch, batch_size, 5))

    # Get all test data
    batch_idx = 0
    for batch_img, batch_voxel in get_while_running(processes[0], queue):
        if batch_idx == num_batch:
            break

        pred, loss, activations = solver.test_output(batch_img, batch_voxel)

        for j in range(batch_size):
            # Save IoU per thresh
            for i, thresh in enumerate(cfg.TEST.VOXEL_THRESH):
                r = evaluate_voxel_prediction(pred[j, ...], batch_voxel[j, ...], thresh)
                results[str(thresh)][batch_idx, j, :] = r

            # Compute AP
            precision = sklearn.metrics.average_precision_score(
                batch_voxel[j, :, 1].flatten(), pred[j, :, 1].flatten())

            results['mAP'][batch_idx, j] = precision

        # record result for the batch
        results['cost'][batch_idx] = float(loss)
        print('%d/%d, costs: %f, mAP: %f' %
                (batch_idx, num_batch, loss, np.mean(results['mAP'][batch_idx])))
        
        batch_idx += 1

    print('Total loss: %f' % np.mean(results['cost']))
    print('Total mAP: %f' % np.mean(results['mAP']))

    sio.savemat(result_fn, results)
