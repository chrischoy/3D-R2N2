import numpy as np
import datetime as dt

# Theano
import theano
import theano.tensor as tensor
from lib.config import cfg

tensor5 = tensor.TensorType(theano.config.floatX, (False,) * 5)


class Net(object):

    def __init__(self, random_seed=dt.datetime.now().microsecond, compute_grad=True):
        self.rng = np.random.RandomState(random_seed)

        self.batch_size = cfg.CONST.BATCH_SIZE
        self.img_w = cfg.CONST.IMG_W
        self.img_h = cfg.CONST.IMG_H
        self.n_vox = cfg.CONST.N_VOX
        self.compute_grad = compute_grad

        # (self.batch_size, 3, self.img_h, self.img_w),
        # override x and is_x_tensor4 when using multi-view network
        self.x = tensor.tensor4()
        self.is_x_tensor4 = True

        # (self.batch_size, self.n_vox, 2, self.n_vox, self.n_vox),
        self.y = tensor5()

        self.activations = []  # list of all intermediate activations
        self.loss = []  # final loss
        self.output = []  # final output
        self.error = []  # final output error
        self.params = []  # all learnable params
        self.grads = []  # will be filled out automatically
        self.setup()

    def setup(self):
        self.network_definition()
        self.post_processing()

    def network_definition(self):
        """ A child network must define
        self.loss
        self.error
        self.params
        self.output
        self.activations is optional
        """
        raise NotImplementedError("Virtual Function")

    def add_layer(self, layer):
        raise NotImplementedError("TODO: add a layer")

    def post_processing(self):
        if self.compute_grad:
            self.grads = tensor.grad(self.loss, [param.val for param in self.params])

    def save(self, filename):
        # params_cpu = {}
        params_cpu = []
        for param in self.params:
            # params_cpu[param.name] = np.array(param.val.get_value())
            params_cpu.append(param.val.get_value())
        np.save(filename, params_cpu)
        print('saving network parameters to ' + filename)

    def load(self, filename, ignore_param=True):
        print('loading network parameters from ' + filename)
        params_cpu_file = np.load(filename)
        if filename.endswith('npz'):
            params_cpu = params_cpu_file[params_cpu_file.keys()[0]]
        else:
            params_cpu = params_cpu_file

        succ_ind = 0
        for param_idx, param in enumerate(self.params):
            try:
                param.val.set_value(params_cpu[succ_ind])
                succ_ind += 1
            except IndexError:
                if ignore_param:
                    print('Ignore mismatch')
                else:
                    raise
