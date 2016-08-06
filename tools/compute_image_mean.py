#!/usr/bin/env python3
'''
Compute image mean in parallel
'''
import _init_paths
import sys
import time
import traceback
import numpy as np

from functools import partial
from itertools import repeat
from lib.config import cfg
from lib.data_io import category_model_id_pair, get_rendering_file
from lib.data_augmentation import image_transform
from PIL import Image

import multiprocessing as mp

train_category_model_pair = category_model_id_pair(dataset_portion=cfg.TRAIN.DATASET_PORTION)


def compute_mean(result_queue, pair):
    try:
        # result_queue, pair = args
        print(pair)
        category, model_id = pair
        accum = np.zeros((127, 127, 3), dtype=np.float64)

        for image_id in range(cfg.TRAIN.NUM_RENDERING):
            image_fn = get_rendering_file(category, model_id, image_id)
            im = Image.open(image_fn)
            t_im = image_transform(np.array(im)[:, :, :3], cfg.TRAIN.PAD_X, cfg.TRAIN.PAD_Y)
            accum += t_im
        result_queue.put(accum / cfg.TRAIN.NUM_RENDERING)
    except:
        traceback.print_exception(*sys.exc_info())
        sys.stdout.flush()

m = mp.Manager()
result_queue = m.Queue(mp.cpu_count() * 2)
pool = mp.Pool(mp.cpu_count())
# jobs = [pair for pair in zip(repeat(result_queue), train_category_model_pair)]
for pair in train_category_model_pair:
    pool.apply_async(compute_mean, args=(result_queue, pair))
pool.close()

# Khan summation
accum = np.zeros((127, 127, 3), dtype=np.float64)
c = np.zeros((127, 127, 3), dtype=np.float64)
tot_count = 0

fail_count = 0
while True:
    if result_queue.qsize() == 0:
        print('Queue empty')
        time.sleep(1)
        fail_count += 1
        if fail_count > 10:
            break
    else:
        try:
            partial_mean = result_queue.get_nowait()
            y = partial_mean - c
            t = accum + y
            c = (t - accum) - y
            accum = t
            fail_count = 0
            tot_count += 1
            accum += partial_mean

            if tot_count % 1000 == 0:
                print('Save {}'.format(tot_count))
                image_mean = accum / tot_count
                np.savez('mean.npz', mean=image_mean)

        except mp.queues.Empty:
            pass

image_mean = accum / tot_count
np.savez('mean.npz', mean=image_mean)
pool.join()

