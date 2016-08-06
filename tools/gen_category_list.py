# Extract a list of main categories (categories that do not have parent categories)
import _init_paths

import json
from config import cfg

categories = json.load(open(cfg.DIR.MODEL_ROOT_PATH + '/taxonomy.json'))

c_dict = {}
for c in categories:
    c_dict[c['synsetId']] = c

for k, v in c_dict.items():
    for c in v['children']:
        if 'parents' not in c_dict[c]:
            c_dict[c]['parents'] = []
        c_dict[c]['parents'].append(k)

out_d = {}
for k, v in c_dict.items():
    if 'parents' not in v: #main categories
        if v['numInstances'] > 1000:
            out_d[k] = {}
            out_d[k]['name'] = v['name']
            out_d[k]['id'] = k
            out_d[k]['dir'] = cfg.DIR.MODEL_ROOT_PATH + '/' + k
            out_d[k]['model_list'] = 'models.txt'

json.dump(out_d, open('./experiments/dataset/shapenet_1000.json.test', 'w+'), indent=4)
