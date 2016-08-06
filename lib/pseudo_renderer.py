#!/usr/bin/env python3

import numpy as np
from PIL import Image
import time
from os import path
from IPython import embed
from lib.data_io import load_model_shapenet
import matplotlib.pyplot as plt
from lib.config import cfg


class PseudoRenderer:
    models_fn =[]
    model_idx = 0
    curr_model_renderings = []
    curr_rendering = ''
    iter_dict = {}

    def __init__(self):
        pass

    def initialize(self, models_fn, viewport_size_x, viewport_size_y):
        self.models_fn = models_fn

    def setViewpoint(self, azimuth, altitude, yaw, distance_ratio, fov):
        if self.model_idx not in self.iter_dict:
            self.iter_dict[self.model_idx] = 0
        self.curr_rendering = self.curr_model_renderings[self.iter_dict[self.model_idx]]
        if self.iter_dict[self.model_idx] + 1 < len(self.curr_model_renderings):
            self.iter_dict[self.model_idx] += 1
        else:
            self.iter_dict[self.model_idx] = 0

    def setModelIndex(self, model_idx):
        self.model_idx = model_idx
        obj_fn = self.models_fn[self.model_idx]
        rendering_dir = path.join(path.dirname(obj_fn), 'rendering')
        rendering_list_fn = path.join(rendering_dir, 'renderings.txt')
        self.curr_model_renderings = [path.join(rendering_dir, line.strip('\n'))
                                    for line in open(rendering_list_fn)]

    def render(self):
        im = np.array(Image.open(self.curr_rendering)) #read as image
        return im[:,:,0:3].transpose((2,1,0)), im[:,:,0]


def main():
    cfg.DATASET = '/cvgl/group/ShapeNet/ShapeNetCore.v1/dataset/cat1000.json'
    models,_ = load_model_shapenet()
    sum_time = 0
    render = PseudoRenderer()
    render.initialize(models, 500, 500)
    for i in range(10000):
        start = time.time()
        render.setModelIndex(0)
        az, el, depth_ratio = list(
            *([360, 5, 0.3] * np.random.rand(1, 3) + [0, 25, 0.65])
        )

        render.setViewpoint(az, el, 0, depth_ratio, 25)
        im,_ = render.render()
        plt.imshow(im.transpose((2,1,0)))
        plt.show()

        end = time.time()
        sum_time += end - start

        if i % 10 == 0:
            print(sum_time/(10))
            sum_time = 0

if __name__ == "__main__":
    main()

