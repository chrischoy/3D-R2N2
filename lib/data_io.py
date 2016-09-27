import os
import json
from collections import OrderedDict

from lib.config import cfg


def id_to_name(id, category_list):
    for k, v in category_list.items():
        if v[0] <= id and v[1] > id:
            return (k, id - v[0])


def category_model_id_pair(dataset_portion=[]):
    '''
    Load category, model names from a shapenet dataset.
    '''

    def model_names(model_path):
        """ Return model names"""
        model_names = [name for name in os.listdir(model_path)
                       if os.path.isdir(os.path.join(model_path, name))]
        return sorted(model_names)

    category_name_pair = []  # full path of the objs files

    cats = json.load(open(cfg.DATASET))
    cats = OrderedDict(sorted(cats.items(), key=lambda x: x[0]))

    for k, cat in cats.items():  # load by categories
        model_path = os.path.join(cfg.DIR.SHAPENET_QUERY_PATH, cat['id'])
        # category = cat['name']
        models = model_names(model_path)
        num_models = len(models)

        portioned_models = models[int(num_models * dataset_portion[0]):int(num_models *
                                                                           dataset_portion[1])]

        category_name_pair.extend([(cat['id'], model_id) for model_id in portioned_models])

    print('lib/data_io.py: model paths from %s' % (cfg.DATASET))

    return category_name_pair


def get_model_file(category, model_id):
    return cfg.DIR.MODEL_PATH % (category, model_id)


def get_voxel_file(category, model_id):
    return cfg.DIR.VOXEL_PATH % (category, model_id)


def get_rendering_file(category, model_id, rendering_id):
    return os.path.join(cfg.DIR.RENDERING_PATH % (category, model_id), '%02d.png' % rendering_id)
