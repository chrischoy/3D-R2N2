#!/usr/bin/env python3

import _init_paths

from lib.data_io import get_voc2012_imglist
from lib.pseudo_renderer import PseudoRenderer

import numpy as np
from PIL import Image

class BackgroundRenderer(PseudoRenderer):
    def initialize(self, models_fn, viewport_size_x, viewport_size_y):
        super().initialize(models_fn, viewport_size_x, viewport_size_y)
        # Read list of Pascal dataset images without overlapping class.
        self.background_imgs = get_voc2012_imglist()

    def render(self):
        train_img = Image.open(self.curr_rendering)
        bg_img = Image.open(np.random.choice(self.background_imgs))

        # Randomly crop background to the size of the training image.
        crop_x = np.random.randint(bg_img.width - train_img.width)
        crop_y = np.random.randint(bg_img.height - train_img.height)
        rendered_img = bg_img.crop((crop_x, crop_y, crop_x + train_img.width,
                                    crop_y + train_img.height))

        # Blend background with given training image.
        rendered_img.paste(train_img, (0, 0), train_img)

        # Return the rendered image in a format compatible with the experiment.
        im = np.array(rendered_img)
        return im.transpose((2, 1, 0)), im[:, :, 0]
