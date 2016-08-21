'''
Demo code for the paper

`Choy et al., 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction, ECCV 2016`
'''
import os
import numpy as np
from subprocess import call

from PIL import Image
from models import load_model
from lib.config import cfg_from_list
from lib.solver import Solver
from lib.voxel import voxel2obj


def download_model(fn):
    if not os.path.isfile(fn):
        # Download the file if doewn't exist
        print('Downloading a pretrained model')
        call(['curl',  'ftp://cs.stanford.edu/cs/cvgl/ResidualGRUNet.npy',
              '--create-dirs', '-o', fn])


def load_demo_images():
    ims = []
    for i in range(3):
        im = Image.open('imgs/%d.png' % i)
        ims.append([np.array(im).transpose((2, 0, 1)).astype(np.float32) / 255.])
    return np.array(ims)


def main():
    '''Main demo function'''
    # Set the batch size to 1
    cfg_from_list(['CONST.BATCH_SIZE', 1])
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
    voxel2obj('prediction.obj', pred[0, :, 1, :, :])
    call(['meshlab', 'prediction.obj'])


if __name__ == '__main__':
    main()
