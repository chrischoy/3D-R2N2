#!/usr/bin/env python
"""
visualize_mesh.py provides functions for mesh visualizations.
"""
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from IPython import embed

def visualize_batch_rendering(rendering):
    plt.imshow((255 * rendering.transpose((1, 2, 0))).astype(np.uint8))


def visualize_batch_predictions(pred, thresh=0.4):
    n_vox, _, _, _ = pred.shape
    visualize_voxels_color_depth(
        pred[:, 1, :, :].reshape(n_vox, n_vox, n_vox),
        thresh, 1, 100, 'Reconstruction, t=%.1f' %(thresh)
    )


def visualize_batch_gt(gt):
    n_vox, _, _, _ = gt.shape
    visualize_voxels_color_depth(
        gt[:, 1, :, :].reshape(n_vox, n_vox, n_vox),
        0.1, 1.1, 100, 'Ground Truth'
    )


def visualize_batch_logic_op(pred, gt, op_name, thresh=0.4):
    pred_occ = pred[:,1,:,:] > thresh
    gt_occ = gt[:,1,:,:]
    if op_name == 'difference':
        op = np.logical_xor
    elif op_name == 'union':
        op = np.logical_or
    elif op_name == 'intersection':
        op = np.logical_and

    op_result = op(pred_occ, gt_occ).astype(int)
    visualize_voxels_color_depth(
        op_result,
        thresh, 1.1, 100, op_name + ' OP, t=' + str(thresh)
    )


def set_axes_range(axes, axes_range):
    axes.set_xlim(axes_range)
    axes.set_ylim(axes_range)
    axes.set_zlim(axes_range)


def visualize_reconstruction(rendering, pred, gt = None, set_ops=[], thresh = 0.4):
    num_subplot = 3 + len(set_ops) if gt is not None else 2
    n_vox,_,_,_ = pred.shape
    plt.clf()
    plt.subplot(1, num_subplot, 1)
    visualize_batch_rendering(rendering)
    axes = plt.gcf().add_subplot(1, num_subplot, 2, projection='3d')
    set_axes_range(axes, [0, n_vox])
    visualize_batch_predictions(pred, thresh)
    if gt is not None:
        axes = plt.gcf().add_subplot(1, num_subplot, 3, projection='3d')
        set_axes_range(axes, [0, n_vox])
        visualize_batch_gt(gt)
        for i, op in enumerate(set_ops):
            axes = plt.gcf().add_subplot(1, num_subplot, 4+i, projection='3d')
            set_axes_range(axes, [0, n_vox])
            visualize_batch_logic_op(pred, gt, op, thresh)
    #plt.show()


def visualize_points(vertices):
    """
    Visualize the input point cloud

    Parameters:
      vertices - point cloud
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2])
    ax.set_aspect('equal')
    plt.show()


def voxel_to_vertices(voxels, min_c=0, max_c=1, n_color=100, return_prob=False):
    """
    Visualize input voxels as point cloud

    Parameters:
      voxels - 3D grid of voxel occupancy map
    """
    # Assume that the input is cubic
    n_shape = voxels.shape[0]
    bins = np.linspace(min_c, max_c, n_color)
    colors = cm.rainbow(bins)

    # bins = np.insert(np.insert(
    #     np.linspace(min_c, max_c, n_color), 0, -np.inf),
    #   n_color + 1, np.inf)
    # colors = cm.rainbow(bins[1:-1])

    n_vert = np.sum(np.logical_and(voxels > min_c, voxels < max_c))
    vertices = np.zeros((n_vert, 3))
    vert_color = np.zeros((n_vert, 4))
    vert_prob = np.zeros((n_vert, 1))

    i_vert = 0
    for i in range(n_shape):
        for j in range(n_shape):
            for k in range(n_shape):
                if voxels[i, j, k] > min_c and voxels[i, j, k] < max_c:
                    vertices[i_vert] = [i, j, k]
                    vert_color[i_vert] = colors[np.digitize([voxels[i, j, k]], bins)]
                    vert_color[i_vert, 3] = voxels[i, j, k]
                    vert_prob[i_vert] = voxels[i, j, k]
                    i_vert += 1

    if return_prob:
        return vertices, vert_color, vert_prob
    else:
        return vertices, vert_color


def visualize_vertices(vertices, colors, ax=None, n_vox=None):
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)


    ax.scatter(vertices[:, 0], vertices[:, 2], vertices[:, 1],
               color=colors, s=20)

    if n_vox is not None:
        ax.set_xlim(0, n_vox)
        ax.set_ylim(0, n_vox)
        ax.set_zlim(0, n_vox)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_zticks([])

    ax.set_aspect('equal')
    # plt.colorbar()


def visualize_voxels_color_depth(voxels, min_c=-1, max_c=2,
                                 n_color=100, title=''):
    """
    Visualize input voxels as point cloud

    Parameters:
      voxels - 3D grid of voxel occupancy map
    """
    # Assume that the input is cubic
    n_shape = voxels.shape[0]
    bins = np.linspace(0, 1.1, n_color)
    colors = cm.rainbow(bins)

    n_vert = np.sum(np.logical_and(voxels > min_c, voxels < max_c))
    vertices = np.zeros((n_vert,3))
    vert_color = np.zeros((n_vert,4))


    i_vert = 0
    for i in range(n_shape):
        for j in range(n_shape):
            for k in range(n_shape):
                if voxels[i, j, k] > min_c and voxels[i, j, k] < max_c:
                    vertices[i_vert] = [i, j, k]
                    i_vert += 1

    # find the spacial extent of the object
    centroid = np.mean(vertices, axis=0)
    max_dist = 0

    viewpoint = np.array([n_shape,0,n_shape])
    ref_pt = viewpoint

    for v1 in vertices:
        if norm(viewpoint-v1) > norm(viewpoint-ref_pt):
            ref_pt = v1
        if norm(centroid-v1) > max_dist:
            max_dist = norm(centroid-v1)
    max_dist *= 2


    for i, v in enumerate(vertices):
        p = np.array(v)
        dist_norm = norm(ref_pt-p)/max_dist
        idx = np.digitize([dist_norm], bins)
        if idx >= colors.shape[0]:
            idx = colors.shape[0]-1
        vert_color[i] = colors[idx]

    ax = plt.gca()
    if title != '':
        ax.set_title(title)

    ax.scatter(vertices[:i_vert, 0],
               vertices[:i_vert, 1],
               vertices[:i_vert, 2],
               color=vert_color)
    ax.set_aspect('equal')


def visualize_rendering(rendering):
    plt.imshow(rendering)
    plt.show()
