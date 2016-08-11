import numpy as np

# Theano
import theano
import theano.tensor as tensor

from models.net import Net, tensor5
from lib.layers import TensorProductLayer, ConvLayer, PoolLayer, Unpool3DLayer, \
    LeakyReLU, SoftmaxWithLoss3D, Conv3DLayer, InputLayer, FlattenLayer, \
    ReshapeLayer, SigmoidLayer, AddLayer, TanhLayer, EltwiseMultiplyLayer, trainable_params


class RecNet(Net):
    def network_definition(self):

        # (multi_views, self.batch_size, 3, self.img_h, self.img_w),
        self.x = tensor5()
        self.is_x_tensor4 = False

        img_w = self.img_w
        img_h = self.img_h
        n_vox_fc = 2
        n_vox_lstm = 8
        # n_vox = self.n_vox

        n_convfilter = [96, 128, 256, 256, 256, 256]
        n_fc_filters = [2048]
        n_deconvfilter = [256, 256, 128, 128, 64, 2]
        input_shape = (self.batch_size, 3, img_w, img_h)

        # To define weights, define the network structure first
        x = InputLayer(input_shape)
        conv1 = ConvLayer(x, (n_convfilter[0], 7, 7))
        pool1 = PoolLayer(conv1)
        conv2 = ConvLayer(pool1, (n_convfilter[1], 3, 3))
        pool2 = PoolLayer(conv2)
        conv3 = ConvLayer(pool2, (n_convfilter[2], 3, 3))
        pool3 = PoolLayer(conv3)
        conv4 = ConvLayer(pool3, (n_convfilter[3], 3, 3))
        pool4 = PoolLayer(conv4)
        conv5 = ConvLayer(pool4, (n_convfilter[4], 3, 3))
        pool5 = PoolLayer(conv5)
        conv6 = ConvLayer(pool5, (n_convfilter[5], 3, 3))
        pool6 = PoolLayer(conv6)
        flat6 = FlattenLayer(pool6)
        fc7   = TensorProductLayer(flat6, n_fc_filters[0])

        # Set the size to be 4 x 4 x 4 x h
        fc_vox_shape = (self.batch_size, n_vox_fc, n_deconvfilter[0], n_vox_fc, n_vox_fc)
        h_vox_shape = (self.batch_size, n_vox_lstm, n_deconvfilter[0], n_vox_lstm, n_vox_lstm)

        # Dummy 3D grid hidden representations
        vox_fc  = ReshapeLayer(fc7, fc_vox_shape[1:])
        unpool8 = Unpool3DLayer(vox_fc)
        conv8   = Conv3DLayer(unpool8, (n_deconvfilter[1], 3, 3, 3))
        unpool9 = Unpool3DLayer(conv8)
        conv9   = Conv3DLayer(unpool9, (n_deconvfilter[2], 3, 3, 3))

        vox_h = InputLayer(h_vox_shape)  # hidden representation
        in_gate_vox_x = Conv3DLayer(conv9, (n_deconvfilter[1], 3, 3, 3))
        in_gate_vox_h = Conv3DLayer(vox_h, (n_deconvfilter[1], 3, 3, 3))

        forget_gate_vox_x = Conv3DLayer(conv9, (n_deconvfilter[1], 3, 3, 3))
        forget_gate_vox_h = Conv3DLayer(vox_h, (n_deconvfilter[1], 3, 3, 3))

        in_transform_vox_x = Conv3DLayer(conv9, (n_deconvfilter[1], 3, 3, 3))
        in_transform_vox_h = Conv3DLayer(vox_h, (n_deconvfilter[1], 3, 3, 3))

        def recurrence(x_curr, prev_vox_h_tensor, prev_vox_c_tensor, prev_vox_in_gate):

            # Scan function cannot use compiled function.
            in_ = InputLayer(input_shape, x_curr)
            conv1_ = ConvLayer(in_, (n_convfilter[0], 7, 7), params=conv1.params)
            pool1_ = PoolLayer(conv1_)
            rect1_ = LeakyReLU(pool1_)
            conv2_ = ConvLayer(rect1_, (n_convfilter[1], 3, 3), params=conv2.params)
            pool2_ = PoolLayer(conv2_)
            rect2_ = LeakyReLU(pool2_)
            conv3_ = ConvLayer(rect2_, (n_convfilter[2], 3, 3), params=conv3.params)
            pool3_ = PoolLayer(conv3_)
            rect3_ = LeakyReLU(pool3_)
            conv4_ = ConvLayer(rect3_, (n_convfilter[3], 3, 3), params=conv4.params)
            pool4_ = PoolLayer(conv4_)
            rect4_ = LeakyReLU(pool4_)
            conv5_ = ConvLayer(rect4_, (n_convfilter[4], 3, 3), params=conv5.params)
            pool5_ = PoolLayer(conv5_)
            rect5_ = LeakyReLU(pool5_)
            conv6_ = ConvLayer(rect5_, (n_convfilter[5], 3, 3), params=conv6.params)
            pool6_ = PoolLayer(conv6_)
            rect6_ = LeakyReLU(pool6_)
            flat6_ = FlattenLayer(rect6_)
            fc7_   = TensorProductLayer(flat6_, n_fc_filters[0], params=fc7.params)
            rect7_ = LeakyReLU(fc7_)

            vox_fc_  = ReshapeLayer(rect7_, fc_vox_shape[1:])
            unpool8_ = Unpool3DLayer(vox_fc_)
            conv8_   = Conv3DLayer(unpool8_, (n_deconvfilter[1], 3, 3, 3), params=conv8.params)
            rect8_   = LeakyReLU(conv8_)
            unpool9_ = Unpool3DLayer(rect8_)
            conv9_   = Conv3DLayer(unpool9_, (n_deconvfilter[2], 3, 3, 3), params=conv9.params)
            rect9_   = LeakyReLU(conv9_)

            vox_h_  = InputLayer(h_vox_shape, prev_vox_h_tensor)
            vox_c_  = InputLayer(h_vox_shape, prev_vox_c_tensor)

            in_gate_vox_x_ = Conv3DLayer(rect9_, (n_deconvfilter[2], 3, 3, 3), params=in_gate_vox_x.params)
            in_gate_vox_h_ = Conv3DLayer(vox_h_, (n_deconvfilter[2], 3, 3, 3), params=in_gate_vox_h.params)

            forget_gate_vox_x_ = Conv3DLayer(rect9_, (n_deconvfilter[2], 3, 3, 3), params=forget_gate_vox_x.params)
            forget_gate_vox_h_ = Conv3DLayer(vox_h_, (n_deconvfilter[2], 3, 3, 3), params=forget_gate_vox_h.params)

            in_transform_vox_x_ = Conv3DLayer(rect9_, (n_deconvfilter[2], 3, 3, 3), params=in_transform_vox_x.params)
            in_transform_vox_h_ = Conv3DLayer(vox_h_, (n_deconvfilter[2], 3, 3, 3), params=in_transform_vox_h.params)

            in_gate_vox_      = SigmoidLayer(AddLayer(in_gate_vox_h_, in_gate_vox_x_))
            forget_gate_vox_  = SigmoidLayer(AddLayer(forget_gate_vox_h_, forget_gate_vox_x_))
            in_transform_vox_ = TanhLayer(AddLayer(in_transform_vox_h_, in_transform_vox_x_))

            c_vox = AddLayer(EltwiseMultiplyLayer(forget_gate_vox_, vox_c_),
                             EltwiseMultiplyLayer(in_gate_vox_, in_transform_vox_))
            h_vox = TanhLayer(c_vox)

            return h_vox.output, c_vox.output, in_gate_vox_.output

        h_c_in_, _ = theano.scan(recurrence,
            sequences=[self.x],  # along with images, feed in the index of the current frame
            outputs_info=
                [tensor.zeros_like(np.zeros(h_vox_shape),
                                   dtype=theano.config.floatX),
                 tensor.zeros_like(np.zeros(h_vox_shape),
                                   dtype=theano.config.floatX),
                 tensor.zeros_like(np.zeros(h_vox_shape),
                                   dtype=theano.config.floatX)])

        in_all = h_c_in_[-1]
        h_vox_all = h_c_in_[0]
        h_vox_last = h_vox_all[-1]

        lstm_h_vox = InputLayer(h_vox_shape, h_vox_last)
        unpool10 = Unpool3DLayer(lstm_h_vox)
        conv10   = Conv3DLayer(unpool10, (n_deconvfilter[3], 3, 3, 3))
        rect10   = LeakyReLU(conv10)
        unpool11 = Unpool3DLayer(rect10)
        conv11   = Conv3DLayer(unpool11, (n_deconvfilter[4], 3, 3, 3))
        rect11   = LeakyReLU(conv11)
        conv12   = Conv3DLayer(rect11, (n_deconvfilter[5], 3, 3, 3))
        softmax_loss = SoftmaxWithLoss3D(conv12.output)

        self.loss = softmax_loss.loss(self.y)
        self.error = softmax_loss.error(self.y)
        self.params = trainable_params
        self.output = softmax_loss.prediction()
        self.activations = [in_all]
