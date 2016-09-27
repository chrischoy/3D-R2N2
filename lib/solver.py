import os
import sys
import theano
import theano.tensor as T
import numpy as np
from datetime import datetime

from lib.config import cfg
from lib.utils import Timer


def max_or_nan(params):
    for param_idx, param in enumerate(params):
        # If there is nan, max will return nan
        nan_or_max_param = np.max(np.abs(param.val.get_value()))
        print('param %d : %f' % (param_idx, nan_or_max_param))
    return nan_or_max_param


def ADAM(lr, params, grads, loss, iteration, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    """
    ADAM update
    """
    t = iteration
    lr_t = lr * T.sqrt(1 - T.pow(beta_2, t)) / (1 - T.pow(beta_1, t))
    w_decay = cfg.TRAIN.WEIGHT_DECAY

    updates = []
    for p, g in zip(params, grads):
        # zero init of moment
        m = theano.shared(p.val.get_value() * 0.)
        # zero init of velocity
        v = theano.shared(p.val.get_value() * 0.)

        if p.is_bias or w_decay == 0:
            regularized_g = g
        else:
            regularized_g = g + w_decay * p.val

        m_t = (beta_1 * m) + (1 - beta_1) * regularized_g
        v_t = (beta_2 * v) + (1 - beta_2) * T.square(regularized_g)
        p_t = p.val - lr_t * m_t / (T.sqrt(v_t) + epsilon)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p.val, p_t))

    return updates


def SGD(lr, params, grads, loss):
    """
    Stochastic Gradient Descent w/ momentum
    """
    momentum = cfg.TRAIN.MOMENTUM
    w_decay = cfg.TRAIN.WEIGHT_DECAY

    updates = []
    for param, grad in zip(params, grads):
        vel = theano.shared(param.val.get_value() * 0.)

        if param.is_bias or w_decay == 0:
            regularized_grad = grad
        else:
            regularized_grad = grad + w_decay * param.val

        param_additive = momentum * vel - lr * regularized_grad
        updates.append((vel, param_additive))
        updates.append((param.val, param.val + param_additive))

    return updates


class Solver(object):

    def __init__(self, net):
        self.net = net
        self.lr = theano.shared(np.float32(1))
        self.iteration = theano.shared(np.float32(0))  # starts from 0
        self._test = None
        self._train_loss = None
        self._test_output = None
        self.compile_model(cfg.TRAIN.POLICY)

    def compile_model(self, policy=cfg.TRAIN.POLICY):
        net = self.net
        lr = self.lr
        iteration = self.iteration

        if policy == 'sgd':
            updates = SGD(lr, net.params, net.grads, net.loss)
        elif policy == 'adam':
            updates = ADAM(lr, net.params, net.grads, net.loss, iteration)
        else:
            sys.exit('Error: Unimplemented optimization policy')

        self.updates = updates

    def set_lr(self, lr):
        self.lr.set_value(lr)

    @property
    def train_loss(self):
        if self._train_loss is None:
            print('Compiling training function')
            self._train_loss = theano.function(
                [self.net.x, self.net.y], self.net.loss, updates=self.updates, profile=cfg.PROFILE)
        self.iteration.set_value(self.iteration.get_value() + 1)
        return self._train_loss

    def train(self, train_queue, val_queue=None):
        ''' Given data queues, train the network '''
        # Parameter directory
        save_dir = os.path.join(cfg.DIR.OUT_PATH)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Timer for the training op and parallel data loading op.
        train_timer = Timer()
        data_timer = Timer()
        training_losses = []

        start_iter = 0
        # Resume training
        if cfg.TRAIN.RESUME_TRAIN:
            self.net.load(cfg.CONST.WEIGHTS)
            start_iter = cfg.TRAIN.INITIAL_ITERATION

        # Setup learning rates
        lr = cfg.TRAIN.DEFAULT_LEARNING_RATE
        lr_steps = [int(k) for k in cfg.TRAIN.LEARNING_RATES.keys()]

        print('Set the learning rate to %f.' % lr)
        self.set_lr(lr)

        # Main training loop
        for train_ind in range(start_iter, cfg.TRAIN.NUM_ITERATION + 1):
            data_timer.tic()
            batch_img, batch_voxel = train_queue.get()
            data_timer.toc()

            if self.net.is_x_tensor4:
                batch_img = batch_img[0]

            # Apply one gradient step
            train_timer.tic()
            loss = self.train_loss(batch_img, batch_voxel)
            train_timer.toc()

            training_losses.append(loss)

            # Decrease learning rate at certain points
            if train_ind in lr_steps:
                # edict only takes string for key. Hacky way
                self.set_lr(np.float(cfg.TRAIN.LEARNING_RATES[str(train_ind)]))
                print('Learing rate decreased to %f: ' % self.lr.get_value())

            # Debugging modules
            #
            # Print status, run validation, check divergence, and save model.
            if train_ind % cfg.TRAIN.PRINT_FREQ == 0:
                # Print the current loss
                print('%s Iter: %d Loss: %f' % (datetime.now(), train_ind, loss))

            if train_ind % cfg.TRAIN.VALIDATION_FREQ == 0 and val_queue is not None:
                # Print test loss and params to check convergence every N iterations
                val_losses = []
                for i in range(cfg.TRAIN.NUM_VALIDATION_ITERATIONS):
                    batch_img, batch_voxel = val_queue.get()
                    _, val_loss, _ = self.test_output(batch_img, batch_voxel)
                    val_losses.append(val_loss)
                print('%s Test loss: %f' % (datetime.now(), np.mean(val_losses)))

            if train_ind % cfg.TRAIN.NAN_CHECK_FREQ == 0:
                # Check that the network parameters are all valid
                max_param = max_or_nan(self.net.params)
                if np.isnan(max_param):
                    print('NAN detected')
                    break

            if train_ind % cfg.TRAIN.SAVE_FREQ == 0 and not train_ind == 0:
                self.save(training_losses, save_dir, train_ind)

            if loss > cfg.TRAIN.LOSS_LIMIT:
                print("Cost exceeds the threshold. Stop training")
                break

    def save(self, training_losses, save_dir, step):
        ''' Save the current network parameters to the save_dir and make a
        symlink to the latest param so that the training function can easily
        load the latest model'''
        save_path = os.path.join(save_dir, 'weights.%d' % (step))
        self.net.save(save_path)

        # Make a symlink for weights.npy
        symlink_path = os.path.join(save_dir, 'weights.npy')
        if os.path.lexists(symlink_path):
            os.remove(symlink_path)

        # Make a symlink to the latest network params
        os.symlink("%s.npy" % os.path.abspath(save_path), symlink_path)

        # Write the losses
        with open(os.path.join(save_dir, 'loss.%d.txt' % step), 'w') as f:
            f.write('\n'.join([str(l) for l in training_losses]))

    def test_output(self, x, y=None):
        '''Generate the reconstruction, loss, and activation. Evaluate loss if
        ground truth output is given. Otherwise, return reconstruction and
        activation'''
        # Cache the output function.
        if self._test_output is None:
            print('Compiling testing function')
            # Lazy load the test function
            self._test_output = theano.function([self.net.x, self.net.y],
                                                [self.net.output,
                                                 self.net.loss,
                                                 *self.net.activations])

        # If the ground truth data is given, evaluate loss. O.w. feed zeros and
        # does not return the loss
        if y is None:
            n_vox = cfg.CONST.N_VOX
            no_loss_return = True
            y_val = np.zeros(
                (cfg.CONST.BATCH_SIZE, n_vox, 2, n_vox, n_vox)).astype(theano.config.floatX)
        else:
            no_loss_return = False
            y_val = y

        # Parse the result
        results = self._test_output(x, y_val)
        prediction = results[0]
        loss = results[1]
        activations = results[2:]

        if no_loss_return:
            return prediction, activations
        else:
            return prediction, loss, activations
