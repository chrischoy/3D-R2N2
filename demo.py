'''
Demo code for the paper

`Choy et al., 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction, ECCV 2016`
'''
import os
import numpy as np
from subprocess import call

from PIL import Image
from models import load_model
from lib.solver import Solver
from lib.voxel import voxel2obj
from lib.data_augmentation import preprocess_img


def download_model(fn):
    if not os.path.isfile(fn):
        # Download the file if doewn't exist
        call('curl ftp://cs.stanford.edu/cs/cvgl/ResidualGRUNet.npy --create-dirs -o %s' % fn)


def load_demo_images():
    ims = []
    for i in range(3):
        im = Image.open('imgs/%d.png' % i)
        ims.append(preprocess_img(im, train=False))
    return np.array(ims)


def main():
    '''Main demo function'''
    # Download pretrained weights
    fn = 'output/ResidualGRUNet/default_model/weights.npy'
    download_model(fn)
    # Use the default network model and instantiate the class
    NetClass = load_model('ResidualGRUNet')
    net = NetClass(compute_grad=False)
    # Load pretrained weights
    net.load(fn)
    # Define a solver. Solver provides wrapper for test functions.
    solver = Solver(net)
    # load images
    demo_imgs = load_demo_images()
    # Run the network
    pred, _ = solver.test_output(demo_imgs)
    # Save the prediction as an OBJ file.
    # Use meshlab or other mesh viewers to visualize the prediction.
    # For Ubuntu>=14.04, you can install meshlab using `sudo apt-get install meshlab`
    voxel2obj('prediction.obj', pred)


if __name__ == '__main__':
    main()
