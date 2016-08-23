import numpy as np

# Theano
import theano
import theano.tensor as tensor

from models.net import Net, tensor5
from lib.layers import TensorProductLayer, ConvLayer, PoolLayer, Unpool3DLayer, \
    LeakyReLU, SoftmaxWithLoss3D, Conv3DLayer, InputLayer, FlattenLayer, \
    FCConv3DLayer, TanhLayer, SigmoidLayer, ComplementLayer, AddLayer, \
    EltwiseMultiplyLayer, get_trainable_params


class ResidualGRUNet(Net):

    def network_definition(self):

        # (multi_views, self.batch_size, 3, self.img_h, self.img_w),
        self.x = tensor5()
        self.is_x_tensor4 = False

        img_w = self.img_w
        img_h = self.img_h
        n_gru_vox = 4
        # n_vox = self.n_vox

        n_convfilter = [96, 128, 256, 256, 256, 256]
        n_fc_filters = [1024]
        n_deconvfilter = [128, 128, 128, 64, 32, 2]
        input_shape = (self.batch_size, 3, img_w, img_h)

        # To define weights, define the network structure first
        x = InputLayer(input_shape)
        conv1a = ConvLayer(x, (n_convfilter[0], 7, 7))
        conv1b = ConvLayer(conv1a, (n_convfilter[0], 3, 3))
        pool1 = PoolLayer(conv1b)

        conv2a = ConvLayer(pool1, (n_convfilter[1], 3, 3))
        conv2b = ConvLayer(conv2a, (n_convfilter[1], 3, 3))
        conv2c = ConvLayer(pool1, (n_convfilter[1], 1, 1))
        pool2 = PoolLayer(conv2c)

        conv3a = ConvLayer(pool2, (n_convfilter[2], 3, 3))
        conv3b = ConvLayer(conv3a, (n_convfilter[2], 3, 3))
        conv3c = ConvLayer(pool2, (n_convfilter[2], 1, 1))
        pool3 = PoolLayer(conv3b)

        conv4a = ConvLayer(pool3, (n_convfilter[3], 3, 3))
        conv4b = ConvLayer(conv4a, (n_convfilter[3], 3, 3))
        pool4 = PoolLayer(conv4b)

        conv5a = ConvLayer(pool4, (n_convfilter[4], 3, 3))
        conv5b = ConvLayer(conv5a, (n_convfilter[4], 3, 3))
        conv5c = ConvLayer(pool4, (n_convfilter[4], 1, 1))
        pool5 = PoolLayer(conv5b)

        conv6a = ConvLayer(pool5, (n_convfilter[5], 3, 3))
        conv6b = ConvLayer(conv6a, (n_convfilter[5], 3, 3))
        pool6 = PoolLayer(conv6b)

        flat6 = FlattenLayer(pool6)
        fc7 = TensorProductLayer(flat6, n_fc_filters[0])

        # Set the size to be 256x4x4x4
        s_shape = (self.batch_size, n_gru_vox, n_deconvfilter[0], n_gru_vox, n_gru_vox)

        # Dummy 3D grid hidden representations
        prev_s = InputLayer(s_shape)

        t_x_s_update = FCConv3DLayer(prev_s, fc7, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3))
        t_x_s_reset = FCConv3DLayer(prev_s, fc7, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3))

        reset_gate = SigmoidLayer(t_x_s_reset)

        rs = EltwiseMultiplyLayer(reset_gate, prev_s)
        t_x_rs = FCConv3DLayer(rs, fc7, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3))

        def recurrence(x_curr, prev_s_tensor, prev_in_gate_tensor):
            # Scan function cannot use compiled function.
            input_ = InputLayer(input_shape, x_curr)
            conv1a_ = ConvLayer(input_, (n_convfilter[0], 7, 7), params=conv1a.params)
            rect1a_ = LeakyReLU(conv1a_)
            conv1b_ = ConvLayer(rect1a_, (n_convfilter[0], 3, 3), params=conv1b.params)
            rect1_ = LeakyReLU(conv1b_)
            pool1_ = PoolLayer(rect1_)

            conv2a_ = ConvLayer(pool1_, (n_convfilter[1], 3, 3), params=conv2a.params)
            rect2a_ = LeakyReLU(conv2a_)
            conv2b_ = ConvLayer(rect2a_, (n_convfilter[1], 3, 3), params=conv2b.params)
            rect2_ = LeakyReLU(conv2b_)
            conv2c_ = ConvLayer(pool1_, (n_convfilter[1], 1, 1), params=conv2c.params)
            res2_ = AddLayer(conv2c_, rect2_)
            pool2_ = PoolLayer(res2_)

            conv3a_ = ConvLayer(pool2_, (n_convfilter[2], 3, 3), params=conv3a.params)
            rect3a_ = LeakyReLU(conv3a_)
            conv3b_ = ConvLayer(rect3a_, (n_convfilter[2], 3, 3), params=conv3b.params)
            rect3_ = LeakyReLU(conv3b_)
            conv3c_ = ConvLayer(pool2_, (n_convfilter[2], 1, 1), params=conv3c.params)
            res3_ = AddLayer(conv3c_, rect3_)
            pool3_ = PoolLayer(res3_)

            conv4a_ = ConvLayer(pool3_, (n_convfilter[3], 3, 3), params=conv4a.params)
            rect4a_ = LeakyReLU(conv4a_)
            conv4b_ = ConvLayer(rect4a_, (n_convfilter[3], 3, 3), params=conv4b.params)
            rect4_ = LeakyReLU(conv4b_)
            pool4_ = PoolLayer(rect4_)

            conv5a_ = ConvLayer(pool4_, (n_convfilter[4], 3, 3), params=conv5a.params)
            rect5a_ = LeakyReLU(conv5a_)
            conv5b_ = ConvLayer(rect5a_, (n_convfilter[4], 3, 3), params=conv5b.params)
            rect5_ = LeakyReLU(conv5b_)
            conv5c_ = ConvLayer(pool4_, (n_convfilter[4], 1, 1), params=conv5c.params)
            res5_ = AddLayer(conv5c_, rect5_)
            pool5_ = PoolLayer(res5_)

            conv6a_ = ConvLayer(pool5_, (n_convfilter[5], 3, 3), params=conv6a.params)
            rect6a_ = LeakyReLU(conv6a_)
            conv6b_ = ConvLayer(rect6a_, (n_convfilter[5], 3, 3), params=conv6b.params)
            rect6_ = LeakyReLU(conv6b_)
            res6_ = AddLayer(pool5_, rect6_)
            pool6_ = PoolLayer(res6_)

            flat6_ = FlattenLayer(pool6_)
            fc7_ = TensorProductLayer(flat6_, n_fc_filters[0], params=fc7.params)
            rect7_ = LeakyReLU(fc7_)

            prev_s_ = InputLayer(s_shape, prev_s_tensor)

            t_x_s_update_ = FCConv3DLayer(
                prev_s_,
                rect7_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3),
                params=t_x_s_update.params)

            t_x_s_reset_ = FCConv3DLayer(
                prev_s_,
                rect7_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3),
                params=t_x_s_reset.params)

            update_gate_ = SigmoidLayer(t_x_s_update_)
            comp_update_gate_ = ComplementLayer(update_gate_)
            reset_gate_ = SigmoidLayer(t_x_s_reset_)

            rs_ = EltwiseMultiplyLayer(reset_gate_, prev_s_)
            t_x_rs_ = FCConv3DLayer(
                rs_, rect7_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3), params=t_x_rs.params)
            tanh_t_x_rs_ = TanhLayer(t_x_rs_)

            gru_out_ = AddLayer(
                EltwiseMultiplyLayer(update_gate_, prev_s_),
                EltwiseMultiplyLayer(comp_update_gate_, tanh_t_x_rs_))

            return gru_out_.output, update_gate_.output

        s_update, _ = theano.scan(recurrence,
            sequences=[self.x],  # along with images, feed in the index of the current frame
            outputs_info=[tensor.zeros_like(np.zeros(s_shape),
                                            dtype=theano.config.floatX),
                           tensor.zeros_like(np.zeros(s_shape),
                                             dtype=theano.config.floatX)])

        update_all = s_update[-1]
        s_all = s_update[0]
        s_last = s_all[-1]
        gru_s = InputLayer(s_shape, s_last)
        unpool7 = Unpool3DLayer(gru_s)
        conv7a = Conv3DLayer(unpool7, (n_deconvfilter[1], 3, 3, 3))
        rect7a = LeakyReLU(conv7a)
        conv7b = Conv3DLayer(rect7a, (n_deconvfilter[1], 3, 3, 3))
        rect7 = LeakyReLU(conv7b)
        res7 = AddLayer(unpool7, rect7)

        unpool8 = Unpool3DLayer(res7)
        conv8a = Conv3DLayer(unpool8, (n_deconvfilter[2], 3, 3, 3))
        rect8a = LeakyReLU(conv8a)
        conv8b = Conv3DLayer(rect8a, (n_deconvfilter[2], 3, 3, 3))
        rect8 = LeakyReLU(conv8b)
        res8 = AddLayer(unpool8, rect8)

        unpool9 = Unpool3DLayer(res8)
        conv9a = Conv3DLayer(unpool9, (n_deconvfilter[3], 3, 3, 3))
        rect9a = LeakyReLU(conv9a)
        conv9b = Conv3DLayer(rect9a, (n_deconvfilter[3], 3, 3, 3))
        rect9 = LeakyReLU(conv9b)

        conv9c = Conv3DLayer(unpool9, (n_deconvfilter[3], 1, 1, 1))
        res9 = AddLayer(conv9c, rect9)

        conv10a = Conv3DLayer(res9, (n_deconvfilter[4], 3, 3, 3))
        rect10a = LeakyReLU(conv10a)
        conv10b = Conv3DLayer(rect10a, (n_deconvfilter[4], 3, 3, 3))
        rect10 = LeakyReLU(conv10b)

        conv10c = Conv3DLayer(rect10a, (n_deconvfilter[4], 3, 3, 3))
        res10 = AddLayer(conv10c, rect10)

        conv11 = Conv3DLayer(res10, (n_deconvfilter[5], 3, 3, 3))
        softmax_loss = SoftmaxWithLoss3D(conv11.output)

        self.loss = softmax_loss.loss(self.y)
        self.error = softmax_loss.error(self.y)
        self.params = get_trainable_params()
        self.output = softmax_loss.prediction()
        self.activations = [update_all]
