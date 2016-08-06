#!/usr/bin/env python3

import _init_paths

import sys
import os
from PIL import Image
from multiprocessing import Process, Event
import numpy as np
import traceback

# Theano
import theano

from lib.config import cfg
from lib.data_augmentation import image_transform, add_random_background, \
    add_random_color_background
from lib.data_io import get_model_file, get_voxel_file, get_rendering_file
from lib.voxel import voxelize_model_binvox

import tools.binvox_rw as binvox_rw


# Force a separate process to print error traces
def print_error(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            traceback.print_exception(*sys.exc_info())
            sys.stdout.flush()
    return func_wrapper


class DataProcess(Process):
    def __init__(self, data_queue, category_model_pair, background_imgs=[]):
        super(DataProcess, self).__init__()
        self.data_queue = data_queue
        self.category_model_pair = category_model_pair
        self.num_data = len(category_model_pair)
        self.batch_size = cfg.CONST.BATCH_SIZE
        self.exit = Event()
        self.shuffle_db_inds()
        self.background_imgs = background_imgs

    def shuffle_db_inds(self):
        # Randomly permute the training roidb
        self.perm = np.random.permutation(np.arange(self.num_data))
        self.cur = 0

    def get_next_minibatch(self):
        if self.cur + self.batch_size >= self.num_data:
            self.shuffle_db_inds()

        db_inds = self.perm[self.cur:self.cur + self.batch_size]
        self.cur += self.batch_size
        return db_inds

    def shutdown(self):
        self.exit.set()

    @print_error
    def run(self):
        # set up constants
        img_h = cfg.CONST.IMG_W
        img_w = cfg.CONST.IMG_H
        n_vox = cfg.CONST.N_VOX

        # This is the maximum number of views
        n_views = cfg.CONST.N_VIEWS

        while not self.exit.is_set():
            # To insure that the network sees (almost) all images per epoch
            db_inds = self.get_next_minibatch()

            # We will sample # views
            if cfg.TRAIN.RANDOM_NUM_VIEWS:
                curr_n_views = np.random.randint(n_views) + 1
            else:
                curr_n_views = n_views

            # This will be fed into the queue. create new batch everytime
            batch_img = np.zeros((curr_n_views, self.batch_size, 3, img_h, img_w),
                                  dtype=theano.config.floatX)
            batch_voxel = np.zeros((self.batch_size, n_vox, 2, n_vox, n_vox),
                                   dtype=theano.config.floatX)

            for batch_id, db_ind in enumerate(db_inds):
                # Data Augmentation.
                category, model_id = self.category_model_pair[db_ind]
                image_ids = np.random.choice(cfg.TRAIN.NUM_RENDERING, curr_n_views)

                for view_id, image_id in enumerate(image_ids):
                    image_fn = get_rendering_file(category, model_id, image_id)
                    im = Image.open(image_fn)

                    # add random background
                    if len(self.background_imgs) > 0:
                        if np.random.rand(1) > cfg.TRAIN.SIMPLE_BACKGROUND_RATIO:
                            im = add_random_background(im, self.background_imgs)
                        else:
                            im = add_random_color_background(im, cfg.TRAIN.NO_BG_COLOR_RANGE)
                    else:
                        # set white background
                        im = add_random_color_background(im, cfg.TRAIN.NO_BG_COLOR_RANGE)

                    im_rgb = np.array(im)[:, :, :3]
                    t_im = image_transform(im_rgb, cfg.TRAIN.PAD_X, cfg.TRAIN.PAD_Y)

                    # Preprocessing
                    if cfg.TRAIN.PREPROCESSING_TYPE == 'center':
                        t_im = t_im - cfg.CONST.IMAGE_MEAN
                    elif cfg.TRAIN.PREPROCESSING_TYPE == 'scale':
                        t_im = t_im / 255.

                    # channel, height, width
                    batch_img[view_id, batch_id, :, :, :] = \
                        t_im.transpose((2, 0, 1))\
                        .astype(theano.config.floatX)

                voxel_fn = get_voxel_file(category, model_id)
                if not os.path.exists(voxel_fn):
                    model_fn = get_model_file(category, model_id)
                    voxel = voxelize_model_binvox(model_fn, cfg.CONST.N_VOX, True)
                    with open(voxel_fn, 'wb') as f:
                        binvox_rw.write(voxel, f)

                with open(voxel_fn, 'rb') as f:
                    voxel = binvox_rw.read_as_3d_array(f)

                voxel_data = voxel.data

                batch_voxel[batch_id, :, 0, :, :] = voxel_data < 1
                batch_voxel[batch_id, :, 1, :, :] = voxel_data

            # The following will wait until the queue frees
            self.data_queue.put((batch_img, batch_voxel), block=True)

        print('Data process ends the while loop')
        return


def test_process():
    from lib.data_io import category_model_id_pair, get_voc2012_imglist
    from multiprocessing import Queue
    from train_net import kill_processes
    from lib.config import cfg

    cfg.TRAIN.PAD_X = 10
    cfg.TRAIN.PAD_Y = 10

    data_queue = Queue(1)
    category_model_pair = category_model_id_pair(dataset_portion=[0, 1])
    voc2012_imglist = get_voc2012_imglist()  # Test random background

    data_process = DataProcess(data_queue, category_model_pair, voc2012_imglist)
    data_process.start()
    batch_img, batch_voxel = data_queue.get()
    import ipdb; ipdb.set_trace()

    kill_processes(data_queue, [data_process])


if __name__ == '__main__':
    test_process()
