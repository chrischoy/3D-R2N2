import inspect
from multiprocessing import Queue

# Training related functions
from models import load_model
from lib.config import cfg
from lib.solver import Solver
from lib.data_io import category_model_id_pair
from lib.data_process import kill_processes, make_data_processes

# Define globally accessible queues, will be used for clean exit when force
# interrupted.
train_queue, val_queue, train_processes, val_processes = None, None, None, None


def cleanup_handle(func):
    '''Cleanup the data processes before exiting the program'''

    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            print('Wait until the dataprocesses to end')
            kill_processes(train_queue, train_processes)
            kill_processes(val_queue, val_processes)
            raise

    return func_wrapper


@cleanup_handle
def train_net():
    '''Main training function'''
    # Set up the model and the solver
    NetClass = load_model(cfg.CONST.NETWORK_CLASS)
    print('Network definition: \n')
    print(inspect.getsource(NetClass.network_definition))
    net = NetClass()

    # Check that single view reconstruction net is not used for multi view
    # reconstruction.
    if net.is_x_tensor4 and cfg.CONST.N_VIEWS > 1:
        raise ValueError('Do not set the config.CONST.N_VIEWS > 1 when using' \
                         'single-view reconstruction network')

    # Generate the solver
    solver = Solver(net)

    # Prefetching data processes
    #
    # Create worker and data queue for data processing. For training data, use
    # multiple processes to speed up the loading. For validation data, use 1
    # since the queue will be popped every TRAIN.NUM_VALIDATION_ITERATIONS.
    global train_queue, val_queue, train_processes, val_processes
    train_queue = Queue(cfg.QUEUE_SIZE)
    val_queue = Queue(cfg.QUEUE_SIZE)

    train_processes = make_data_processes(
        train_queue,
        category_model_id_pair(dataset_portion=cfg.TRAIN.DATASET_PORTION),
        cfg.TRAIN.NUM_WORKER,
        repeat=True)
    val_processes = make_data_processes(
        val_queue,
        category_model_id_pair(dataset_portion=cfg.TEST.DATASET_PORTION),
        1,
        repeat=True,
        train=False)

    # Train the network
    solver.train(train_queue, val_queue)

    # Cleanup the processes and the queue.
    kill_processes(train_queue, train_processes)
    kill_processes(val_queue, val_processes)


def main():
    '''Test function'''
    cfg.DATASET = '/cvgl/group/ShapeNet/ShapeNetCore.v1/cat1000.json'
    cfg.CONST.RECNET = 'rec_net'
    cfg.TRAIN.DATASET_PORTION = [0, 0.8]
    train_net()


if __name__ == '__main__':
    main()
