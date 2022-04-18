import os
import os.path as osp
from glob import glob
import json
from argparse import ArgumentParser 

def main(args):
    anno_path = args.anno_dir
    annos = glob(osp.join(anno_path, '*.json'))
    print(annos)

    annotations = dict()
    annotations['images'] = dict()
    for anno in annos:
        with open(anno, 'r') as f:
            data_json = json.loads(f.read())
            annotations['images'].update(data_json['images'])
    
    with open(osp.join(anno_path, 'train.json'), 'w') as f:
        json.dump(annotations, f, indent=4)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--anno_dir', type=str,
                        default="/opt/ml/input/data/ICDAR17_Korean/ufo")
    args = parser.parse_args()
    main(args)