import os
import numpy as np
import matplotlib.pyplot as plt
import importlib

# Theano & network
import theano
from lib.config import cfg
from lib.solver import Solver
from lib.data_augmentation import add_random_color_background, crop_center
from lib.data_io import get_model_file, get_voxel_file, \
    get_rendering_file, category_model_id_pair
from lib.visualize_mesh import visualize_reconstruction

import tools.binvox_rw as binvox_rw
from lib.voxel import voxelize_model_binvox, evaluate_voxel_prediction
from PIL import Image
import scipy.io

from IPython import embed


def test_net():
    result_dir = os.path.join(cfg.DIR.OUT_PATH, cfg.TEST.EXP_NAME)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    result_fn = os.path.join(result_dir, 'result.mat')

    print("Exp file will be written to: " + result_fn)

    # load weights
    netlib= importlib.import_module("models.%s" % (cfg.CONST.RECNET))
    net = netlib.RecNet(compute_grad=False)
    net_fn = os.path.join(cfg.CONST.WEIGHTS)
    net.load(net_fn)
    solver = Solver(net)
    # set constants
    batch_size = cfg.CONST.BATCH_SIZE
    img_h = cfg.CONST.IMG_W
    img_w = cfg.CONST.IMG_H
    n_vox = cfg.CONST.N_VOX
    n_views = cfg.CONST.N_VIEWS

    # set up testing data
    category_model_pair = \
        category_model_id_pair(dataset_portion=cfg.TEST.DATASET_PORTION)

    num_model = len(category_model_pair)
    num_batch = int(num_model / batch_size)
    model_ids = np.random.choice(num_model, num_model, False)
    # prepare result container

    results = {}
    results['cost'] = np.zeros(num_batch)
    for thresh in cfg.TEST.VOXEL_THRESH:
        results[str(thresh)] = np.zeros((num_batch, batch_size, 5))

    if cfg.TEST.VISUALIZE:
        fig = plt.gcf()
        fig.set_size_inches(40, 10)

    for batch_idx in range(num_batch):
        db_inds = model_ids[batch_idx*batch_size:(batch_idx+1)*batch_size]

        # We will sample # views
        curr_n_views = n_views
        # if cfg.TRAIN.RANDOM_NUM_VIEWS:
        #     curr_n_views = np.random.randint(n_views) + 1

        # This will be fed into the queue. create new batch everytime
        batch_img = np.zeros((curr_n_views, batch_size, 3, img_h, img_w),
                              dtype=theano.config.floatX)
        batch_voxel = np.zeros((batch_size, n_vox, 2, n_vox, n_vox),
                               dtype=theano.config.floatX)

        for datum_id_idx, datum_id in enumerate(db_inds):
            # Data Augmentation.
            category, model_id = category_model_pair[datum_id]
            # image_ids = np.random.choice(cfg.TRAIN.NUM_RENDERING, curr_n_views)
            image_ids = np.arange(n_views)

            for view_id, image_id in enumerate(image_ids):
                image_fn = get_rendering_file(category, model_id, image_id)
                im = Image.open(image_fn)

                # set white background
                im = add_random_color_background(im, cfg.TEST.NO_BG_COLOR_RANGE)

                im_rgb = np.array(im)[:, :, :3]
                t_im = crop_center(im_rgb, img_h, img_w)

                # channel, height, width
                batch_img[view_id, datum_id_idx, :, :, :] = \
                    t_im.transpose((2, 0, 1))\
                    .astype(theano.config.floatX) / 255.

            voxel_fn = get_voxel_file(category, model_id)
            if not os.path.exists(voxel_fn):
                model_fn = get_model_file(category, model_id)
                voxel = voxelize_model_binvox(model_fn, cfg.CONST.N_VOX, True)
                with open(voxel_fn, 'wb') as f:
                    binvox_rw.write(voxel, f)

            with open(voxel_fn, 'rb') as f:
                voxel = binvox_rw.read_as_3d_array(f)

            voxel_data = voxel.data

            batch_voxel[datum_id_idx, :, 0, :, :] = voxel_data < 1
            batch_voxel[datum_id_idx, :, 1, :, :] = voxel_data

        pred, loss, activations = solver.test_output(batch_img, batch_voxel)
        print ('%d/%d, cost is: %f' %(batch_idx, num_batch, loss))

        # record result for the batch
        results['cost'][batch_idx] = float(loss)
        for i, thresh in enumerate(cfg.TEST.VOXEL_THRESH):
            for j in range(batch_size):
                r = evaluate_voxel_prediction(pred[j, ...], batch_voxel[j, ...], thresh)
                results[str(thresh)][batch_idx, j, :] = r

        if cfg.TEST.VISUALIZE:
            for thresh in cfg.TEST.VOXEL_THRESH:
                visualize_reconstruction(batch_img[0,0,...],
                        pred[0,...].transpose(0, 1, 3, 2),
                        batch_voxel[0,...].transpose(0, 1, 3, 2),
                        ['intersection'], thresh)
                fig.subtitle('cost=%.3f' % (float(loss)), fontsize=24)
                im_fn = os.path.join(result_dir, '%i.pdf' % batch_idx)
                plt.savefig(im_fn)

    print('Total loss: %f' % np.mean(results['cost']))
    scipy.io.savemat(result_fn, results)
