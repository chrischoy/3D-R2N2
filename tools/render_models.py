#!/usr/bin/env python3

import _init_paths
import numpy as np
import os

from lib.config import cfg
from lib.data_io import category_model_id_pair, get_model_file
from multiprocessing import Pool


def render_model(category, model_id, viewpoints_num, overwrite=False):
    from lib.blender_renderer import ShapeNetRenderer
    """Render a single model using a given renderer."""
    model_paths = get_model_file(category, model_id)
    rendering_dir = os.path.join(cfg.DIR.RENDERING_ROOT_PATH, category,
                                 model_id, cfg.DIR.RENDERING_SUBFOLDER)

    if not os.path.isdir(rendering_dir):
        os.makedirs(rendering_dir)

    # Rendering results configuration.
    rendered_img_list = os.path.join(rendering_dir,
                                     cfg.RENDERING.RENDERED_IMG_LIST)
    rendered_img_metadata = os.path.join(rendering_dir,
                                         cfg.RENDERING.RENDERED_IMG_METADATA)
    if os.path.isfile(rendered_img_list) and not overwrite:
        return

    # Load renderer.
    if cfg.RENDERING.RENDER_ENGINE == 'blender':
        renderer = ShapeNetRenderer()
    else:
        raise NotImplementedError('Unsupported rendering engine: ' +
                                  cfg.RENDERING.RENDER_ENGINE)

    # Initialize renderer and load model.
    viewport_size_x = cfg.CONST.IMG_W + cfg.TRAIN.CROP_X
    viewport_size_y = cfg.CONST.IMG_H + cfg.TRAIN.CROP_Y
    renderer.initialize([model_paths], viewport_size_x, viewport_size_y)
    renderer.loadModel()

    # Render images from random viewpoints.
    view_configs = []
    rendered_filenames = []
    for i in range(viewpoints_num):
        # Set random viewpoint.
        az, el, depth_ratio = list(
            *([360, 5, 0.3] * np.random.rand(1, 3) + [0, 25, 0.65]))
        renderer.setViewpoint(az, el, 0, depth_ratio, 25)

        # Render an image of a given viewpoint.
        rendered_filename = '%02d.png' % i
        renderer.result_fn = os.path.join(rendering_dir, rendered_filename)

        rendering, depth = renderer.render(load_model=False, clear_model=False)

        if rendering is None:
            raise RuntimeError("Rendering failed for unknown reason.")

        # Save the viewpoint.
        rendered_filenames.append(rendered_filename)
        view_configs.append([str(az), str(el), '0', str(depth_ratio), '25'])

    renderer.clearModel()

    # Write rendered image list with its metadata.
    with open(rendered_img_list, 'w') as f:
        for rendered_filename in rendered_filenames:
            f.write(rendered_filename + '\n')

    with open(rendered_img_metadata, 'w') as f:
        for view_config in view_configs:
            f.write(' '.join(view_config) + '\n')


def main():
    cfg.TRAIN.CROP_X = 10
    cfg.TRAIN.CROP_Y = 10

    NUM_IMG = cfg.TRAIN.NUM_RENDERING
    OUTPUT_DIR = 'rendering'  # Subfolder per model to save prerendered images.

    # Multi-process pre-rendering
    # Blender tends to get slower after it renders few hundred models. Start
    # over the whole pool every BATCH_SIZE models to boost the speed.
    NUM_PROCESS = 6
    BATCH_SIZE = 200
    args = [(category, model_id, NUM_IMG) for category, model_id in
            category_model_id_pair(dataset_portion=[0, 1])]

    args_batches = [args[i * BATCH_SIZE:min((i + 1) * BATCH_SIZE, len(args))]
                    for i in range(len(args) // BATCH_SIZE + 1)]

    for args_batch in args_batches:
        with Pool(processes=NUM_PROCESS) as pool:
            pool.starmap(render_model, args_batch)

if __name__ == '__main__':
    main()
