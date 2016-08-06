import _init_paths
import os
import numpy as np

from lib.utils import stdout_redirected
from tempfile import TemporaryFile

import tools.binvox_rw as binvox_rw


def evaluate_voxel_prediction(preds, gt, thresh):
    preds_occupy = preds[:, 1, :, :] >= thresh
    diff = np.sum(np.logical_xor(preds_occupy, gt[:, 1, :, :]))
    intersection = np.sum(np.logical_and(preds_occupy, gt[:, 1, :, :]))
    union = np.sum(np.logical_or(preds_occupy, gt[:, 1, :, :]))
    num_fp = np.sum(np.logical_and(preds_occupy, gt[:, 0, :, :]))  # false positive
    num_fn = np.sum(np.logical_and(np.logical_not(preds_occupy), gt[:, 1, :, :]))  # false negative
    return np.array([diff, intersection, union, num_fp, num_fn])


def voxelize_model_binvox(obj, n_vox, return_voxel=True, binvox_add_param=''):
    cmd = "./tools/binvox -d %d -cb -dc -aw -pb %s -t binvox %s" % (
            n_vox, binvox_add_param, obj)

    if not os.path.exists(obj):
        raise ValueError('No obj found : %s' % obj)

    # Stop printing command line output
    with TemporaryFile() as f, stdout_redirected(f):
        os.system(cmd)

    # load voxelized model
    if return_voxel:
        with open('%s.binvox' % obj[:-4], 'rb') as f:
            vox = binvox_rw.read_as_3d_array(f)

        return vox.data
