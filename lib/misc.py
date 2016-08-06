import numpy as np


def softmax3D(voxels):
    """
    Given 2 channel prediction per voxel,
    return probability distribution for each channel
    """
    exp_voxels = np.exp(voxels)
    n_vox, n_ch, _,  _ = exp_voxels.shape
    return exp_voxels / np.sum(exp_voxels, axis=1, keepdims=True)
