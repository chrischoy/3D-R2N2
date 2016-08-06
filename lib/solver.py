import numpy as np
import theano
import theano.tensor as T
from lib.config import cfg
import sys


def ADAM(lr, params, grads, loss, iteration,
         beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    """
    ADAM update
    """
    t = iteration
    lr_t = lr * T.sqrt(1 - T.pow(beta_2, t)) / (1 - T.pow(beta_1, t))
    w_decay = cfg.TRAIN.WEIGHT_DECAY

    updates = []
    for p, g in zip(params, grads):
        # zero init of moment
        m = theano.shared(p.val.get_value()*0.)
        # zero init of velocity
        v = theano.shared(p.val.get_value()*0.)

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
        vel = theano.shared(param.val.get_value()*0.)

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

    def compile_model(self, policy=cfg.TRAIN.POLICY):
        net = self.net
        lr = self.lr
        iteration = self.iteration

        if policy == 'sgd':
            updates = SGD(lr, net.params, net.grads, net.loss)
        # elif cfg.TRAIN.POLICY == 'rmsprop':
        #     updates = RMSProp(lr, net.params, net.grads, net.loss)
        # elif cfg.TRAIN.POLICY == 'adadelta':
        #     updates = AdaDelta(lr, net.params_, net.grads, net.loss)
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
            self._train_loss = theano.function([self.net.x, self.net.y],
                                               self.net.loss,
                                               updates=self.updates,
                                               profile=cfg.PROFILE)
        self.iteration.set_value(self.iteration.get_value() + 1)
        return self._train_loss

    def test_output(self, x, y=None):
        """Test function"""
        if self._test_output is None:
            print('Compiling testing function')
            self._test_output = theano.function([self.net.x, self.net.y],
                                                [self.net.output, self.net.loss])

        if y is None:
            n_vox = cfg.CONST.N_VOX
            no_loss_return = True
            y_val = np.zeros((n_vox, n_vox, n_vox)).astype(theano.config.floatX)
        else:
            no_loss_return = False
            y_val = y
        results = self._test_output(x, y_val)
        prediction = results[0]
        loss = results[1]
        activations = results[2:]
        if no_loss_return:
            return prediction, activations
        else:
            return prediction, loss, activations

