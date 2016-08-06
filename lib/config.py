import numpy as np
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Common
#
__C.SUB_CONFIG_FILE = []
__C.DATASET = '/cvgl/group/ShapeNet/ShapeNetCore.v1/cat1000.json'  # yaml/json file that specifies a dataset (training/testing)
__C.NET_NAME = 'reconstruction_net'
__C.PROFILE = False

__C.CONST = edict()
__C.CONST.DEVICE = 'gpu0'
__C.CONST.RNG_SEED = 0
__C.CONST.IMG_W = 127
__C.CONST.IMG_H = 127
__C.CONST.N_VOX = 32
__C.CONST.N_VIEWS = 1
__C.CONST.BATCH_SIZE = 36
__C.CONST.RECNET = 'rec_net'
__C.CONST.WEIGHTS = '/scratch/chrischoy/mv_deep_res_gru_net_3x3x3/max_5_views_no_rnd_bg/weights.npy.39999.npy'  # when set, load the weights from the file
__C.CONST.IMAGE_MEAN = np.load('mean.npz')['mean']

#
# Directories
#
__C.DIR = edict()
# Path where taxonomy.json is stored
__C.DIR.MODEL_ROOT_PATH = '/cvgl/group/ShapeNet/ShapeNetCore.v1/'
__C.DIR.MODEL_PATH = '/cvgl/group/ShapeNet/ShapeNetCore.v1/%s/%s/model.obj'
__C.DIR.VOXEL_PATH = '/cvgl/group/ShapeNet/ShapeNetCore.v1/%s/%s/model.binvox'
__C.DIR.RENDERING_PATH = '/cvgl/group/ShapeNet/ShapeNetCore.v1/%s/%s/rendering'
__C.DIR.OUT_PATH = './output/default'

#
# Training
#
__C.TRAIN = edict()

__C.TRAIN.RESUME_TRAIN = False
__C.TRAIN.INITIAL_ITERATION = 30000  # when the training resumes, set the iteration number
__C.TRAIN.USE_REAL_IMG = False
__C.TRAIN.DATASET_PORTION = [0, 0.8]

# Data worker
__C.TRAIN.NUM_WORKER = 1  # number of data workers
__C.TRAIN.NUM_ITERATION = 120000  # maximum number of training iterations
__C.TRAIN.WORKER_LIFESPAN = 100  # if use blender, kill a worker after some iteration to clear cache
__C.TRAIN.WORKER_CAPACITY = 1000  # if use OSG, load only limited number of models at a time
__C.TRAIN.NUM_RENDERING = 24
__C.TRAIN.NUM_VALIDATION_ITERATIONS = 24
__C.TRAIN.RANDOM_NUM_VIEWS = True  # feed in random # views if n_views > 1

__C.QUEUE_SIZE = 15  # maximum number of minibatches that can be put in a data queue

# Data augmentation
__C.TRAIN.RANDOM_CROP = True
__C.TRAIN.PAD_X = 10
__C.TRAIN.PAD_Y = 10
__C.TRAIN.HUE_CHANGE = True
__C.TRAIN.FLIP = True
__C.TRAIN.HUE_RANGE = 0.1

# For no random bg images, add random colors
__C.TRAIN.NO_BG_COLOR_RANGE = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.RANDOM_BACKGROUND = False
__C.TRAIN.SIMPLE_BACKGROUND_RATIO = 0.5  # ratio of the simple backgrounded images

# Learning
# For SGD use 0.1, for ADAM, use 0.0001
__C.TRAIN.DEFAULT_LEARNING_RATE = 1e-4
__C.TRAIN.POLICY = 'adam'  # def: sgd, adam
# The EasyDict can't use dict with integers as keys
__C.TRAIN.LEARNING_RATES = {'40000': 1e-5,
                            '80000': 1e-6}
__C.TRAIN.MOMENTUM = 0.95
# weight decay or regularization constant. If not set, the loss can diverge
# after the training almost converged since weight can increase indefinitely
# (for cross entropy loss). Too high regularization will also hinder training.
__C.TRAIN.WEIGHT_DECAY = 0.00005

__C.TRAIN.SAVE_FREQ = 10000   # weights will be overwritten every save_freq
__C.TRAIN.PRINT_FREQ = 40

###############################################################################
# WARNING
# You cannot use the scale trained model for centered model
# Scale: scale 1/255
# Center: subtract mean from the image
###############################################################################
__C.TRAIN.PREPROCESSING_TYPE = 'center' # ['scale', 'center']

#
# Testing options
#
__C.TEST = edict()
__C.TEST.EXP_NAME = 'test'
__C.TEST.USE_IMG = False
__C.TEST.MODEL_ID = []
__C.TEST.DATASET_PORTION = [0.8, 1]
__C.TEST.SAMPLE_SIZE = 0
__C.TEST.IMG_PATH = ''
__C.TEST.AZIMUTH = []
__C.TEST.NO_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]

__C.TEST.VISUALIZE = False
__C.TEST.VOXEL_THRESH = [0.4]

#
# Rendering
#
__C.RENDERING = edict()
__C.RENDERING.RENDERING_ENGINE = 'blender'  # ['blender', 'osg', 'prerender']
__C.RENDERING.USE_PRERENDERING = True
__C.RENDERING.MAX_CAMERA_DIST = 1.75
__C.RENDERING.RENDERED_IMG_LIST = 'renderings.txt'  # File where list of rendered images will be written.
__C.RENDERING.RENDERED_IMG_METADATA = 'rendering_metadata.txt'  # File where metadata of corresponding rendered images will be written.
__C.RENDERING.BLENDER_TMP_DIR = '/tmp/'


#
# Pascal dataset
#
__C.PASCAL = edict()
__C.PASCAL.VOC2012_DIR = '/cvgl/group/VOC2012'  # Directory with VOC2012 dataset.
__C.PASCAL.BLACKLIST_CLASSES = ('aeroplane', 'boat', 'bus', 'car', 'chair',
                                'diningtable', 'sofa', 'tvmonitor')  # Classes to avoid.
__C.PASCAL.CLASSES_DIR = 'ImageSets/Main'  # Directory with image class informations.
__C.PASCAL.IMGS_DIR = 'JPEGImages'  # Directory with datset images.
# Files with information about image classes.
__C.PASCAL.CLASSES_FILES = ('_val.txt', '_train.txt', '_trainval.txt')

__C.PASCAL3D = edict()
# Classes to use for evaluation. Comes from CategoryShape.
__C.PASCAL3D.EVAL_CLASSES = ('aeroplane', 'bicycle', 'boat', 'bus', 'car',
                             'chair', 'motorbike', 'sofa', 'train', 'tvmonitor')
# Directory with voxelized PASCAL3D.
__C.PASCAL3D.VOXEL_DIR = '/cvgl/group/3DEverything/evaluation/pascal3d_voxel'
# File with train/test metadata of PASCAL3D, just as in akar43/CategoryShapes.
__C.PASCAL3D.EVAL_METADATA = '/cvgl/group/3DEverything/evaluation/pascal_metadata.mat'
# Directory with PASCAL3D+_release1.0 annotation. 2.0 doesn't work.
__C.PASCAL3D.ANNOTATION = '/cvgl/group/3DEverything/evaluation/PASCAL3D+_release1.0/Annotations'

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b.keys():
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
