import os
import json
import scipy.io
from collections import OrderedDict

from lib.config import cfg


def id_to_name(id, category_list):
    for k, v in category_list.items():
        if v[0] <= id and v[1] > id:
            return (k, id - v[0])


def return_aligned_models(model_path, model_ids, num_models):
    """ Load metadata"""

    def orientation_mapping(x):
        """Used for Seeing 3D Chair dataset json metadata loading"""
        return {
            'left': 0,
            'right': 180,
            'up': 90,
            'up_left': 45,
            'up_right': 135,
            'down': -90,
            'down_left': -45,
            'down_right': -135
        }[x]

    aligned_models = []
    for model_id in model_ids:
        metadata_file = open(os.path.join(model_path, model_id,
                                          "metadata.json"))
        model_metadata = json.load(metadata_file)
        metadata_file.close()

        azimuth_offset = orientation_mapping(model_metadata['orientation'])
        if azimuth_offset == 0:
            aligned_models.append(model_id)
            if len(aligned_models) == num_models:
                return aligned_models


def category_model_id_pair(dataset_portion=[]):
    '''
    Load category, model names from a shapenet dataset.
    '''
    def model_names(model_path, model_file):
        """ Return model names"""
        model_names = [line.rstrip('\n') for line in
                       open(os.path.join(model_path, model_file), 'r')]
        return model_names

    # model_ids = range(model_ids[0], model_ids[1]) if model_ids else None
    # use_portion_only = not model_ids
    category_name_pair = []  # full path of the objs files

    cats = json.load(open(cfg.DATASET))
    cats = OrderedDict(sorted(cats.items(), key=lambda x: x[0]))

    for k, cat in cats.items():  # load by categories
        model_file = cat['model_list']
        model_path = cat['dir']
        # category_id = cat['id'] or k
        # category = cat['name']

        models = model_names(model_path, model_file)
        num_models = len(models)

        portioned_models = models[
            int(num_models * dataset_portion[0]):
            int(num_models * dataset_portion[1])
        ]

        category_name_pair.extend([(cat['id'], model_id) for model_id in portioned_models])

    print('lib/data_io.py: model paths from %s' % (cfg.DATASET))

    return category_name_pair


def get_model_file(category, model_id):
    return cfg.DIR.MODEL_PATH % (category, model_id)


def get_voxel_file(category, model_id):
    return cfg.DIR.VOXEL_PATH % (category, model_id)


def get_rendering_file(category, model_id, rendering_id):
    return os.path.join(cfg.DIR.RENDERING_PATH % (category, model_id),
                        '%02d.png' % rendering_id)


def get_voc2012_imglist():
    """Retrieves list of PASCAL image that can be used for random background."""
    whitelist_img = set()  # Set of class-safe images to return.
    blacklist_img = set()
    classes_path = os.path.join(cfg.PASCAL.VOC2012_DIR, cfg.PASCAL.CLASSES_DIR)
    # Parse all class definition files for each class.
    for c in cfg.PASCAL.BLACKLIST_CLASSES:
        for file_name in cfg.PASCAL.CLASSES_FILES:
            class_file = os.path.join(classes_path, c + file_name)
            with open(class_file) as f:
                for line in f.readlines():
                    image_file, class_exists = line.rstrip().split()
                    # Add image_file to whitelist if it doesn't have any of the
                    # blacklisted class in it.
                    if class_exists == '-1' and image_file not in blacklist_img:
                        whitelist_img.add(image_file)
                    else:
                        whitelist_img.discard(image_file)
                        blacklist_img.add(image_file)
    # Return full path of whitelisted image files.
    return [os.path.join(cfg.PASCAL.VOC2012_DIR, cfg.PASCAL.IMGS_DIR,
                         img + '.jpg') for img in whitelist_img]


def get_voc2012_eval_metadata(is_train=True):
    """Retrieves PASCAL dataset evaluation metadata.

    Returns tuple of ('data', 'label').
    'label' is an integer vector indexing 'classes'.
    'data' is a struct array with following perperties:
    ['imsize', 'voc_image_id', 'voc_rec_id', 'pascal_bbox', 'view', 'kps',
     'part_names', 'bbox', 'poly_x', 'poly_y', 'class', 'flip', 'rotP3d',
     'euler', 'subtype', 'objectIndP3d']
    """
    metadata = scipy.io.loadmat(cfg.PASCAL3D.EVAL_METADATA, squeeze_me=True)
    if is_train:
        return (metadata['train_data'], metadata['train_label'])
    else:
        return (metadata['test_data'], metadata['test_label'])
