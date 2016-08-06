import _init_paths

import sys
import os
import json
import traceback
from multiprocessing import Process

from lib.data_io import get_model_file, get_voxel_file
from lib.config import cfg
from lib.voxel import voxelize_model_binvox

USE_XVFP_SERVER = False
USE_SUBPROCESS = True
N_VOX = 64
OVERWRITE = True


def voxelize_model_subprocess(category_id, model_list_path, n_vox=N_VOX):

    model_ids = [line.rstrip('\n') for line in
                   open(os.path.join(model_list_path, 'models.txt'), 'r')]

    for i, model_id in enumerate(model_ids):
        model_fn = get_model_file(category_id, model_id)
        print('voxelizing %d/%d: %s' % (i+1, len(model_ids), model_fn))
        if not OVERWRITE and os.path.exists(model_fn):
            print('Already voxelized, skipping')
            continue

        sys.stdout.flush()  # To push print while running inside a Process
        voxelize_model_binvox(model_fn, n_vox, return_voxel=False)


def main():
    cats = json.load(open(cfg.DATASET))

    pl = []
    # Use binvox server

    if USE_XVFP_SERVER:
        os.system('Xvfb :99 -screen 0 640x480x24 &')
        os.system('export DISPLAY=:99')

    # setup
    for category_id, cat in cats.items():
        model_list_path = cat['dir']
        if USE_SUBPROCESS:
            p = Process(target=voxelize_model_subprocess,
                        args=(category_id, model_list_path))
            p.start()
            pl.append(p)
        else:
            voxelize_model_subprocess(category_id, model_list_path)

    if USE_SUBPROCESS:
        for p in pl:
            p.join()


if __name__ == '__main__':
    main()
