import os
import os.path as osp
import json
import numpy as np

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default="/opt/ml/input/data/ICDAR17_Korean/images")
    parser.add_argument('--anno_dir', type=str,
                        default="/opt/ml/input/data/ICDAR17_Korean/ufo/")
    parser.add_argument('--save_dir', type=str,
                        default="/opt/ml/input/data/ICDAR17_Korean/pre_images")
    parser.add_argument('--image_size', type=int, default=1024)

    args = parser.parse_args()
    return args

def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x)

def is_valid_image_size(x, image_size):
    if x >= image_size:
        return True
    return False

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

def main(data_dir, anno_dir, save_dir, image_size):
    maybe_mkdir(save_dir)

    data = read_json(osp.join(anno_dir, 'train.json'))
    new_json = dict()
    new_json['images'] = dict()
    total_num = len(data['images'])

    with tqdm(total=total_num) as pbar:
        for image_key, image_value in data['images'].items():            
            img_w = image_value['img_w']
            img_h = image_value['img_h']
            ratio = image_size / max(img_w, img_h)

            if not(is_valid_image_size(img_w, image_size) and is_valid_image_size(img_h, image_size)):
                continue
            
            words = image_value['words']
            for word_idx in range(len(words)):
                image_value['words'][str(word_idx)]['points'] =\
                    (np.array(image_value['words'][str(word_idx)]['points']) * np.array(ratio)).tolist()

            if img_w > img_h:
                image_value['img_w'] = image_size
                image_value['img_h'] = int(img_h * ratio)
            else:
                image_value['img_w'] = int(img_w * ratio)
                image_value['img_h'] = image_size

            new_json['images'].update({image_key:image_value})

            img = Image.open(osp.join(data_dir, image_key)).convert('RGB')
            img = img.resize((image_value['img_w'], image_value['img_h']), Image.ANTIALIAS)
            img.save(
                osp.join(save_dir, image_key), quality=100)
            
            pbar.update(1)

    with open(osp.join(anno_dir, 'new_train.json'), 'w') as f:
        json.dump(new_json, f, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)
