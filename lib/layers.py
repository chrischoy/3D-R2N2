import numpy as np

# Theano
import collections
import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv, conv3d2d, sigmoid
from theano.tensor.signal import pool

trainable_params = []


def get_trainable_params():
    global trainable_params
    return trainable_params


class Weight(object):

    def __init__(self,
                 w_shape,
                 is_bias,
                 mean=0,
                 std=0.01,
                 filler='msra',
                 fan_in=None,
                 fan_out=None,
                 name=None):
        super(Weight, self).__init__()
        assert (is_bias in [True, False])
        rng = np.random.RandomState()

        if isinstance(w_shape, collections.Iterable) and not is_bias:
            if len(w_shape) > 1 and len(w_shape) < 5:
                fan_in = np.prod(w_shape[1:])
                fan_out = np.prod(w_shape) / w_shape[1]
                n = (fan_in + fan_out) / 2.
            elif len(w_shape) == 5:
                # 3D Convolution filter
                fan_in = np.prod(w_shape[1:])
                fan_out = np.prod(w_shape) / w_shape[2]
                n = (fan_in + fan_out) / 2.
            else:
                raise NotImplementedError(
                    'Filter shape with ndim > 5 not supported: len(w_shape) = %d' % len(w_shape))
        else:
            n = 1

        if fan_in and fan_out:
            n = (fan_in + fan_out) / 2.

        if filler == 'gaussian':
            self.np_values = np.asarray(rng.normal(mean, std, w_shape), dtype=theano.config.floatX)
        elif filler == 'msra':
            self.np_values = np.asarray(
                rng.normal(mean, np.sqrt(2. / n), w_shape), dtype=theano.config.floatX)
        elif filler == 'xavier':
            scale = np.sqrt(3. / n)
            self.np_values = np.asarray(
                rng.uniform(
                    low=-scale, high=scale, size=w_shape), dtype=theano.config.floatX)
        elif filler == 'constant':
            self.np_values = np.cast[theano.config.floatX](mean * np.ones(
                w_shape, dtype=theano.config.floatX))
        elif filler == 'orth':
            ndim = np.prod(w_shape)
            W = np.random.randn(ndim, ndim)
            u, s, v = np.linalg.svd(W)
            self.np_values = u.astype(theano.config.floatX).reshape(w_shape)
        else:
            raise NotImplementedError('Filler %s not implemented' % filler)

        self.is_bias = is_bias  # Either the weight is bias or not
        self.val = theano.shared(value=self.np_values)
        self.shape = w_shape
        self.name = name

        global trainable_params
        trainable_params.append(self)


class InputLayer(object):

    def __init__(self, input_shape, tinput=None):
        self._output_shape = input_shape
        self._input = tinput

    @property
    def output(self):
        if self._input is None:
            raise ValueError('Cannot call output for the layer. Initialize' \
                             + ' the layer with an input argument')
        return self._input

    @property
    def output_shape(self):
        return self._output_shape


class Layer(object):
    ''' Layer abstract class. support basic functionalities.
    If you want to set the output shape, either prev_layer or input_shape must
    be defined.

    If you want to use the computation graph, provide either prev_layer or set_input
    '''

    def __init__(self, prev_layer):
        self._output = None
        self._output_shape = None
        self._prev_layer = prev_layer
        self._input_shape = prev_layer.output_shape
        # Define self._output_shape

    def set_output(self):
        '''Override the function'''
        # set self._output using self._input=self._prev_layer.output
        raise NotImplementedError('Layer virtual class')

    @property
    def output_shape(self):
        if self._output_shape is None:
            raise ValueError('Set output shape first')
        return self._output_shape

    @property
    def output(self):
        if self._output is None:
            self.set_output()
        return self._output


class TensorProductLayer(Layer):

    def __init__(self, prev_layer, n_out, params=None, bias=True):
        super().__init__(prev_layer)
        self._bias = bias
        n_in = self._input_shape[-1]

        if params is None:
            self.W = Weight((n_in, n_out), is_bias=False)
            if bias:
                self.b = Weight((n_out,), is_bias=True, mean=0.1, filler='constant')
        else:
            self.W = params[0]
            if bias:
                self.b = params[1]

        # parameters of the model
        self.params = [self.W]
        if bias:
            self.params.append(self.b)

        self._output_shape = [self._input_shape[0]]
        self._output_shape.extend(self._input_shape[1:-1])
        self._output_shape.append(n_out)

    def set_output(self):
        self._output = tensor.dot(self._prev_layer.output, self.W.val)
        if self._bias:
            self._output += self.b.val


class BlockDiagonalLayer(Layer):
    """
    Compute block diagonal matrix multiplication efficiently using broadcasting

    Last dimension will be used for matrix multiplication.

    prev_layer.output_shape = N x D_1 x D_2 x ... x D_{n-1} x D_n
    output_shape            = N x D_1 x D_2 x ... x D_{n-1} x n_out
    """

    def __init__(self, prev_layer, n_out, params=None, bias=True):
        super().__init__(prev_layer)
        self._bias = bias
        self._output_shape = list(self._input_shape)
        self._output_shape[-1] = n_out
        self._output_shape = tuple(self._output_shape)

        if params is None:
            self._W_shape = list(self._input_shape[1:])
            self._W_shape.append(n_out)
            self._W_shape = tuple(self._W_shape)
            self.W = Weight(self._W_shape, is_bias=False)
            if bias:
                self.b = Weight(self._output_shape[1:], is_bias=True, mean=0.1, filler='constant')
        else:
            self.W = params[0]
            if bias:
                self.b = params[1]

        # parameters of the model
        self.params = [self.W]
        if bias:
            self.params.append(self.b)

    def set_output(self):
        self._output = tensor.sum(tensor.shape_padright(self._prev_layer.output) *
                                  tensor.shape_padleft(self.W.val),
                                  axis=-2)
        if self._bias:
            self._output += tensor.shape_padleft(self.b.val)


class AddLayer(Layer):

    def __init__(self, prev_layer, add_layer):
        super().__init__(prev_layer)
        self._output_shape = self._input_shape
        self._add_layer = add_layer

    def set_output(self):
        self._output = self._prev_layer.output + self._add_layer.output


class EltwiseMultiplyLayer(Layer):

    def __init__(self, prev_layer, mult_layer):
        super().__init__(prev_layer)
        self._output_shape = self._input_shape
        self._mult_layer = mult_layer

    def set_output(self):
        self._output = self._prev_layer.output * self._mult_layer.output


class FlattenLayer(Layer):

    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self._output_shape = [self._input_shape[0], np.prod(self._input_shape[1:])]

    def set_output(self):
        self._output = \
            self._prev_layer.output.flatten(2)  # flatten from the second dim


class DimShuffleLayer(Layer):

    def __init__(self, prev_layer, shuffle_pattern):
        super().__init__(prev_layer)
        self._shuffle_pattern = shuffle_pattern
        self._output_shape = list(self._input_shape)
        for out_dim, in_dim in enumerate(shuffle_pattern):
            self._output_shape[out_dim] = self._input_shape[in_dim]
        self._output_shape = tuple(self._output_shape)

    def set_output(self):
        self._output = self._prev_layer.output.dimshuffle(self._shuffle_pattern)


class ReshapeLayer(Layer):

    def __init__(self, prev_layer, reshape):
        super().__init__(prev_layer)
        self._output_shape = [self._prev_layer.output_shape[0]]
        self._output_shape.extend(reshape)
        self._output_shape = tuple(self._output_shape)
        print('Reshape the prev layer to [%s]' % ','.join(str(x) for x in self._output_shape))

    def set_output(self):
        self._output = tensor.reshape(self._prev_layer.output, self._output_shape)


class ConvLayer(Layer):
    """Conv Layer
    filter_shape: [n_out_channel, n_height, n_width]

    self._input_shape: [batch_size, n_in_channel, n_height, n_width]
    """

    def __init__(self, prev_layer, filter_shape, padding=True, params=None):
        super().__init__(prev_layer)
        self._padding = padding
        self._filter_shape = [filter_shape[0], self._input_shape[1], filter_shape[1],
                              filter_shape[2]]
        if params is None:
            self.W = Weight(self._filter_shape, is_bias=False)
            self.b = Weight((filter_shape[0],), is_bias=True, mean=0.1, filler='constant')
        else:
            for i, s in enumerate(self._filter_shape):
                assert (params[0].shape[i] == s)
            self.W = params[0]
            self.b = params[1]

        self.params = [self.W, self.b]

        # Define self._output_shape
        if padding and filter_shape[1] * filter_shape[2] > 1:
            self._padding = [0, 0, int((filter_shape[1] - 1) / 2), int((filter_shape[2] - 1) / 2)]
            self._output_shape = [self._input_shape[0], filter_shape[0], self._input_shape[2],
                                  self._input_shape[3]]
        else:
            self._padding = [0] * 4
            # TODO: for the 'valid' convolution mode the following is the
            # output shape. Diagnose failure
            self._output_shape = [self._input_shape[0], filter_shape[0],
                                  self._input_shape[2] - filter_shape[1] + 1,
                                  self._input_shape[3] - filter_shape[2] + 1]

    def set_output(self):
        if sum(self._padding) > 0:
            padded_input = tensor.alloc(0.0,  # Value to fill the tensor
                                        self._input_shape[0],
                                        self._input_shape[1],
                                        self._input_shape[2] + 2 * self._padding[2],
                                        self._input_shape[3] + 2 * self._padding[3])

            padded_input = tensor.set_subtensor(
                padded_input[:, :, self._padding[2]:self._padding[2] + self._input_shape[2],
                             self._padding[3]:self._padding[3] + self._input_shape[3]],
                self._prev_layer.output)

            padded_input_shape = [self._input_shape[0], self._input_shape[1],
                                  self._input_shape[2] + 2 * self._padding[2],
                                  self._input_shape[3] + 2 * self._padding[3]]
        else:
            padded_input = self._prev_layer.output
            padded_input_shape = self._input_shape

        conv_out = tensor.nnet.conv2d(
            input=padded_input,
            filters=self.W.val,
            filter_shape=self._filter_shape,
            input_shape=padded_input_shape,
            border_mode='valid')

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self._output = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')


class PoolLayer(Layer):

    def __init__(self, prev_layer, pool_size=(2, 2), padding=(1, 1)):
        super().__init__(prev_layer)
        self._pool_size = pool_size
        self._padding = padding
        img_rows = self._input_shape[2] + 2 * padding[0]
        img_cols = self._input_shape[3] + 2 * padding[1]
        out_r = (img_rows - pool_size[0]) // pool_size[0] + 1
        out_c = (img_cols - pool_size[1]) // pool_size[1] + 1
        self._output_shape = [self._input_shape[0], self._input_shape[1], out_r, out_c]

    def set_output(self):
        pooled_out = pool.pool_2d(
            input=self._prev_layer.output,
            ds=self._pool_size,
            ignore_border=True,
            padding=self._padding)
        self._output = pooled_out


class Unpool3DLayer(Layer):
    """3D Unpooling layer for a convolutional network """

    def __init__(self, prev_layer, unpool_size=(2, 2, 2), padding=(0, 0, 0)):
        super().__init__(prev_layer)
        self._unpool_size = unpool_size
        self._padding = padding
        output_shape = (self._input_shape[0],  # batch
                        unpool_size[0] * self._input_shape[1] + 2 * padding[0],  # depth
                        self._input_shape[2],  # out channel
                        unpool_size[1] * self._input_shape[3] + 2 * padding[1],  # row
                        unpool_size[2] * self._input_shape[4] + 2 * padding[2])  # col
        self._output_shape = output_shape

    def set_output(self):
        output_shape = self._output_shape
        padding = self._padding
        unpool_size = self._unpool_size
        unpooled_output = tensor.alloc(0.0,  # Value to fill the tensor
                                       output_shape[0],
                                       output_shape[1] + 2 * padding[0],
                                       output_shape[2],
                                       output_shape[3] + 2 * padding[1],
                                       output_shape[4] + 2 * padding[2])

        unpooled_output = tensor.set_subtensor(unpooled_output[:, padding[0]:output_shape[
            1] + padding[0]:unpool_size[0], :, padding[1]:output_shape[3] + padding[1]:unpool_size[
                1], padding[2]:output_shape[4] + padding[2]:unpool_size[2]],
                                               self._prev_layer.output)
        self._output = unpooled_output


class Conv3DLayer(Layer):
    """3D Convolution layer"""

    def __init__(self, prev_layer, filter_shape, padding=None, params=None):
        super().__init__(prev_layer)
        self._filter_shape = [filter_shape[0],  # out channel
                              filter_shape[1],  # time
                              self._input_shape[2],  # in channel
                              filter_shape[2],  # height
                              filter_shape[3]]  # width
        self._padding = padding

        # signals: (batch,       in channel, depth_i, row_i, column_i)
        # filters: (out channel, in channel, depth_f, row_f, column_f)

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        if params is None:
            self.W = Weight(self._filter_shape, is_bias=False)
            self.b = Weight((filter_shape[0],), is_bias=True, mean=0.1, filler='constant')
            params = [self.W, self.b]
        else:
            self.W = params[0]
            self.b = params[1]

        self.params = [self.W, self.b]

        if padding is None:
            self._padding = [0, int((filter_shape[1] - 1) / 2), 0, int((filter_shape[2] - 1) / 2),
                             int((filter_shape[3] - 1) / 2)]

        self._output_shape = [self._input_shape[0], self._input_shape[1], filter_shape[0],
                              self._input_shape[3], self._input_shape[4]]

    def set_output(self):
        padding = self._padding
        input_shape = self._input_shape
        if np.sum(self._padding) > 0:
            padded_input = tensor.alloc(0.0,  # Value to fill the tensor
                                        input_shape[0],
                                        input_shape[1] + 2 * padding[1],
                                        input_shape[2],
                                        input_shape[3] + 2 * padding[3],
                                        input_shape[4] + 2 * padding[4])

            padded_input = tensor.set_subtensor(
                padded_input[:, padding[1]:padding[1] + input_shape[1], :, padding[3]:padding[3] +
                             input_shape[3], padding[4]:padding[4] + input_shape[4]],
                self._prev_layer.output)
        else:
            padded_input = self._prev_layer.output

        self._output = conv3d2d.conv3d(padded_input, self.W.val) + \
            self.b.val.dimshuffle('x', 'x', 0, 'x', 'x')


class FCConv3DLayer(Layer):
    """3D Convolution layer with FC input and hidden unit"""

    def __init__(self, prev_layer, fc_layer, filter_shape, padding=None, params=None):
        """Prev layer is the 3D hidden layer"""
        super().__init__(prev_layer)
        self._fc_layer = fc_layer
        self._filter_shape = [filter_shape[0],  # out channel
                              filter_shape[2],  # time
                              filter_shape[1],  # in channel
                              filter_shape[3],  # height
                              filter_shape[4]]  # width
        self._padding = padding

        if padding is None:
            self._padding = [0, int((self._filter_shape[1] - 1) / 2), 0, int(
                (self._filter_shape[3] - 1) / 2), int((self._filter_shape[4] - 1) / 2)]

        self._output_shape = [self._input_shape[0], self._input_shape[1], filter_shape[0],
                              self._input_shape[3], self._input_shape[4]]

        if params is None:
            self.Wh = Weight(self._filter_shape, is_bias=False)

            self._Wx_shape = [self._fc_layer._output_shape[1], np.prod(self._output_shape[1:])]

            # Each 3D cell will have independent weights but for computational
            # speed, we expand the cells and compute a matrix multiplication.
            self.Wx = Weight(
                self._Wx_shape,
                is_bias=False,
                fan_in=self._input_shape[1],
                fan_out=self._output_shape[2])

            self.b = Weight((filter_shape[0],), is_bias=True, mean=0.1, filler='constant')
            params = [self.Wh, self.Wx, self.b]
        else:
            self.Wh = params[0]
            self.Wx = params[1]
            self.b = params[2]

        self.params = [self.Wh, self.Wx, self.b]

    def set_output(self):
        padding = self._padding
        input_shape = self._input_shape
        padded_input = tensor.alloc(0.0,  # Value to fill the tensor
                                    input_shape[0],
                                    input_shape[1] + 2 * padding[1],
                                    input_shape[2],
                                    input_shape[3] + 2 * padding[3],
                                    input_shape[4] + 2 * padding[4])

        padded_input = tensor.set_subtensor(padded_input[:, padding[1]:padding[1] + input_shape[
            1], :, padding[3]:padding[3] + input_shape[3], padding[4]:padding[4] + input_shape[4]],
                                            self._prev_layer.output)

        fc_output = tensor.reshape(
            tensor.dot(self._fc_layer.output, self.Wx.val), self._output_shape)
        self._output = conv3d2d.conv3d(padded_input, self.Wh.val) + \
            fc_output + self.b.val.dimshuffle('x', 'x', 0, 'x', 'x')


class Conv3DLSTMLayer(Layer):
    """Convolution 3D LSTM layer

    Unlike a standard LSTM cell witch doesn't have a spatial information,
    Convolutional 3D LSTM has limited connection that respects spatial
    configuration of LSTM cells.

    The filter_shape defines the size of neighbor that the 3D LSTM cells will consider.
    """

    def __init__(self, prev_layer, filter_shape, padding=None, params=None):

        super().__init__(prev_layer)
        prev_layer._input_shape
        n_c = filter_shape[0]
        n_x = self._input_shape[2]
        n_neighbor_d = filter_shape[1]
        n_neighbor_h = filter_shape[2]
        n_neighbor_w = filter_shape[3]

        # Compute all gates in one convolution
        self._gate_filter_shape = [4 * n_c, 1, n_x + n_c, 1, 1]

        self._filter_shape = [filter_shape[0],  # num out hidden representation
                              filter_shape[1],  # time
                              self._input_shape[2],  # in channel
                              filter_shape[2],  # height
                              filter_shape[3]]  # width
        self._padding = padding

        # signals: (batch,       in channel, depth_i, row_i, column_i)
        # filters: (out channel, in channel, depth_f, row_f, column_f)

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        if params is None:
            self.W = Weight(self._filter_shape, is_bias=False)
            self.b = Weight((filter_shape[0],), is_bias=True, mean=0.1, filler='constant')
            params = [self.W, self.b]
        else:
            self.W = params[0]
            self.b = params[1]

        self.params = [self.W, self.b]

        if padding is None:
            self._padding = [0, int((filter_shape[1] - 1) / 2), 0, int((filter_shape[2] - 1) / 2),
                             int((filter_shape[3] - 1) / 2)]

        self._output_shape = [self._input_shape[0], self._input_shape[1], filter_shape[0],
                              self._input_shape[3], self._input_shape[4]]

    def set_output(self):
        padding = self._padding
        input_shape = self._input_shape
        padded_input = tensor.alloc(0.0,  # Value to fill the tensor
                                    input_shape[0],
                                    input_shape[1] + 2 * padding[1],
                                    input_shape[2],
                                    input_shape[3] + 2 * padding[3],
                                    input_shape[4] + 2 * padding[4])

        padded_input = tensor.set_subtensor(padded_input[:, padding[1]:padding[1] + input_shape[
            1], :, padding[3]:padding[3] + input_shape[3], padding[4]:padding[4] + input_shape[4]],
                                            self._prev_layer.output)

        self._output = conv3d2d.conv3d(padded_input, self.W.val) + \
            self.b.val.dimshuffle('x', 'x', 0, 'x', 'x')


class SoftmaxWithLoss3D(object):
    """
    Softmax with loss (n_batch, n_vox, n_label, n_vox, n_vox)
    """

    def __init__(self, input):
        self.input = input
        self.exp_x = tensor.exp(self.input)
        self.sum_exp_x = tensor.sum(self.exp_x, axis=2, keepdims=True)

    def prediction(self):
        return self.exp_x / self.sum_exp_x

    def error(self, y, threshold=0.5):
        return tensor.mean(tensor.eq(tensor.ge(self.prediction(), threshold), y))

    def loss(self, y):
        """
        y must be a tensor that has the same dimensions as the input. For each
        channel, only one element is one indicating the ground truth prediction
        label.
        """
        return tensor.mean(
            tensor.sum(-y * self.input, axis=2, keepdims=True) + tensor.log(self.sum_exp_x))


class ConcatLayer(Layer):

    def __init__(self, prev_layers, axis=1):
        """
        list of prev layers to concatenate
        axis to concatenate

        For tensor5, channel dimension is axis=2 (due to theano conv3d
        convention). For image, axis=1
        """
        assert (len(prev_layers) > 1)
        super().__init__(prev_layers[0])
        self._axis = axis
        self._prev_layers = prev_layers

        self._output_shape = self._input_shape.copy()
        for prev_layer in prev_layers[1:]:
            self._output_shape[axis] += prev_layer._output_shape[axis]
        print('Concat the prev layer to [%s]' % ','.join(str(x) for x in self._output_shape))

    def set_output(self):
        self._output = tensor.concatenate([x.output for x in self._prev_layers], axis=self._axis)


class LeakyReLU(Layer):

    def __init__(self, prev_layer, leakiness=0.01):
        super().__init__(prev_layer)
        self._leakiness = leakiness
        self._output_shape = self._input_shape

    def set_output(self):
        self._input = self._prev_layer.output
        if self._leakiness:
            # The following is faster than T.maximum(leakiness * x, x),
            # and it works with nonsymbolic inputs as well. Also see:
            # http://github.com/benanne/Lasagne/pull/163#issuecomment-81765117
            f1 = 0.5 * (1 + self._leakiness)
            f2 = 0.5 * (1 - self._leakiness)
            self._output = f1 * self._input + f2 * abs(self._input)
            # self.param = [self.leakiness]
        else:
            self._output = 0.5 * (self._input + abs(self._input))


class SigmoidLayer(Layer):

    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self._output_shape = self._input_shape

    def set_output(self):
        self._output = sigmoid(self._prev_layer.output)


class TanhLayer(Layer):

    def __init__(self, prev_layer):
        super().__init__(prev_layer)

    def set_output(self):
        self._output = tensor.tanh(self._prev_layer.output)


class ComplementLayer(Layer):
    """ Compute 1 - input_layer.output """

    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self._output_shape = self._input_shape

    def set_output(self):
        self._output = tensor.ones_like(self._prev_layer.output) - self._prev_layer.output
