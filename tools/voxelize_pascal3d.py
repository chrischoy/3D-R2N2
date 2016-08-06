import _init_paths

import os
import glob

from lib.config import cfg
from lib.voxel import voxelize_model_binvox

def voxelize_pascal3d():
    """Voxelize PASCAL3D models for evaluation.

    Rotates voxel to match axis with that of ShapeNet. Must be rotated back when
    running evaluation.
    """
    for c in cfg.PASCAL3D.EVAL_CLASSES:
        voxel_dir = cfg.PASCAL3D.VOXEL_DIR
        for off_file in glob.iglob(os.path.join(voxel_dir, c, '*.off')):
            voxelize_model_binvox(off_file, cfg.CONST.N_VOX, return_voxel=False,
                    binvox_add_param='-rotx -rotx -rotx -ary 270')

if __name__ == '__main__':
    voxelize_pascal3d()
