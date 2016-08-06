import _init_paths

from lib.blender_renderer import VoxelRenderer
from lib.voxel2obj import voxel2mesh, write_obj
from lib.config import cfg
import tools.binvox_rw as binvox_rw
from PIL import Image
import os
import numpy as np
import glob
import scipy.io


def get_pascal3d_filelist(base_dir, ext='*', classes=None):
    """Get a list of files from PASCAL 3D+ file structure."""
    files = []
    classes = classes or cfg.PASCAL3D.EVAL_CLASSES
    for c in classes:
        files.extend(glob.glob(os.path.join(os.path.join(base_dir, c), ext)))
    return files


def parse_result(result_f=None):
    if result_f is None:
        result_f = os.path.join(cfg.DIR.OUT_PATH, cfg.TEST.EXP_NAME,
                'result.mat')
    results = scipy.io.loadmat(result_f, squeeze_me=True)
    scores = results['pred'].transpose(0, 1, 3, 2, 4, 5)[:, :, 1, :, :, :]
    scores = scores.reshape(scores.shape[0] * scores.shape[1], scores.shape[2],
                            scores.shape[3], scores.shape[4])
    meta = results.get('meta', np.array([])).flatten()
    if 'input' in results:
        img = results['input']
        img = img.reshape(img.shape[0] * img.shape[1], img.shape[2],
                          img.shape[3], img.shape[4])
        img = img.transpose(0, 2, 3, 1)
    else:
        img = None
    return scores, meta, img


def objfy_binvox(binvox_list):
    """Get a list of full path of binvox files and write obj files."""
    for f in binvox_list:
        with open(f, 'rb') as fp:
            voxels = binvox_rw.read_as_3d_array(fp).data
        out_f = os.path.join(os.path.splitext(f)[0] + '.obj')
        vertices, faces = voxel2mesh(voxels)
        write_obj(out_f, vertices, faces)


def objfy_scores(scores, base_dir, file_list=None, threshold=0.1):
    """Get a list of full path of binvox files and write obj files."""
    num_results = scores.shape[0]
    if file_list is None:
        file_list = [str(i) for i in range(num_results)]
    for i in range(num_results):
        out_f = os.path.join(base_dir, file_list[i] + '_%d.obj' % i)
        vertices, faces = voxel2mesh(scores[i])
        write_obj(out_f, vertices, faces)


def save_input_imgs(imgs, base_dir, file_list=None):
    """Get a list of full path of binvox files and write obj files."""
    num_imgs = imgs.shape[0]
    if file_list is None:
        file_list = [str(i) for i in range(num_imgs)]
    for i in range(num_imgs):
        out_f = os.path.join(base_dir, file_list[i] + '_%d.png' % i)
        Image.fromarray(np.uint8(imgs[i] * 255)).save(out_f)


def render_objs(obj_list, viewpoint, render_append_fn='', render_size=500):
    """Get a list of full path of obj files and render images."""
    renderer = VoxelRenderer()
    renderer.initialize(obj_list, render_size, render_size)
    for i, obj_f in enumerate(obj_list):
        img_f = os.path.splitext(obj_f)[0] + render_append_fn + '.png'
        renderer.setModelIndex(i)
        renderer.setViewpoint(*viewpoint)
        rendering, alpha = renderer.render(load_model=True, clear_model=True,
                                           image_path=img_f)


def main():
    VIEW_POINT = (45, 30, 0, 0.6, 25)
    SCORE_OUT_DIR = '/scr/jgwak/SourceCodes/3deverything/output/voxels'
    IMG_OUT_DIR = '/scr/jgwak/SourceCodes/3deverything/output/imgs'
    CATEGORY_SHAPE_VOX_DIR = '/scr/jgwak/SourceCodes/3deverything-external/CategoryShapes/meshWithKps'

    # Category Shape visualization.
    # Category shape result binvox to obj.
    binvox_list =  get_pascal3d_filelist(CATEGORY_SHAPE_VOX_DIR, ext='*_pred.binvox', classes=['boat'])
    objfy_binvox(binvox_list)
    # Category Shape obj to png.
    obj_list =  get_pascal3d_filelist(CATEGORY_SHAPE_VOX_DIR, ext='*_pred.obj', classes=['boat'])
    render_objs(obj_list, VIEW_POINT)

    # Our result visualization.
    # Test result to obj.
    scores, meta, img = parse_result()
    objfy_scores(scores, SCORE_OUT_DIR, meta)
    # Test result obj to png.
    obj_list = glob.glob(os.path.join(SCORE_OUT_DIR, '*.obj'))
    render_objs(obj_list, VIEW_POINT)
    # Save test input images.
    save_input_imgs(img, IMG_OUT_DIR, meta)


if __name__ == "__main__":
    main()
