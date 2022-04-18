import json
import os
import os.path as osp
from glob import glob
from PIL import Image
from argparse import ArgumentParser 

import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, ConcatDataset, Dataset


SRC_DATASET_DIR = '/opt/ml/input/data/ICDAR17_MLT'  # FIXME
DST_DATASET_DIR = '/opt/ml/input/data/ICDAR17_Korean'  # FIXME

NUM_WORKERS = 32  # FIXME

IMAGE_EXTENSIONS = {'.gif', '.jpg', '.png'}

LANGUAGE_MAP = {
    'Korean': 'ko',
    'Latin': 'en',
    'Symbols': None
}

def get_language_token(x):
    return LANGUAGE_MAP.get(x, 'others')


def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x)


class MLT17Dataset(Dataset):
    def __init__(self, image_dir, label_dir, copy_images_to=None):
        image_paths = {x for x in glob(osp.join(image_dir, '*')) if osp.splitext(x)[1] in
                       IMAGE_EXTENSIONS}

        label_path = glob(osp.join(label_dir, '*.json'))[0]
        label_gts = []
        with open(label_path, 'r') as f:
            label_gts.append(json.loads(f.read()))
        
        assert len(image_paths) == len(label_gts[0])

        sample_ids, samples_info = list(), dict()
        for image_path in image_paths:
            sample_id = osp.splitext(osp.basename(image_path))[0]

            label_gt = label_gts[0][sample_id]

            words_info, extra_info = self.parse_label_file(label_gt)
            if extra_info['languages'].difference({'ko', 'en'}):
                continue
            print(sample_id)
            sample_ids.append(sample_id)
            samples_info[sample_id] = dict(image_path=image_path,
            words_info=words_info)

        self.sample_ids, self.samples_info = sample_ids, samples_info

        self.copy_images_to = copy_images_to

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_info = self.samples_info[self.sample_ids[idx]]

        image_fname = osp.basename(sample_info['image_path'])
        image = Image.open(sample_info['image_path'])
        img_w, img_h = image.size

        if self.copy_images_to:
            maybe_mkdir(self.copy_images_to)
            image.save(osp.join(self.copy_images_to, osp.basename(sample_info['image_path'])))

        license_tag = dict(usability=True, public=True, commercial=True, type='CC-BY-SA',
                           holder=None)
        sample_info_ufo = dict(img_h=img_h, img_w=img_w, words=sample_info['words_info'], tags=None,
                               license_tag=license_tag)

        return image_fname, sample_info_ufo

    def parse_label_file(self, label_gt):
        def rearrange_points(points):
            start_idx = np.argmin([np.linalg.norm(p, ord=1) for p in points])
            if start_idx != 0:
                points = np.roll(points, -start_idx, axis=0).tolist()
            return points

        def polygon_to_rectangle(points, transcription):
            points = np.array(points)
            words, rects = [], []
            for idx in range(len(points)//2 -1):
                ltx, lty = points[idx, 0], points[idx, 1]
                rtx, rty = points[idx+1, 0], points[idx+1, 1]
                rbx, rby = points[-idx-2, 0], points[-idx-2, 1]
                lbx, lby = points[-idx-1, 0], points[-idx-1, 1]
                rects.append([[ltx, lty], [rtx, rty], [rbx, rby], [lbx, lby]])
                if transcription == "Unknown":
                    words.append("Unknown")
                    continue
                
                if transcription == "###":
                    words.append("###")
                    continue

                words.append([
                    words[idx * len(transcription)//(len(points)//2-1):(idx + 1)* len(transcription)//(len(points)//2-1)]])

            return rects, words

        words_info, languages = dict(), set()
        word_idx = 0
        for items in label_gt:
            language, transcription = items['language'], items['transcription']
            points = items['points']

            if len(points) < 4:
                continue

            if len(points) == 4:
                points = np.array(points, dtype=np.float32).reshape(4, 2).tolist()
                points = rearrange_points(points)

                illegibility = transcription == '###'
                orientation = 'Horizontal'
                language = get_language_token(language)
                words_info[word_idx] = dict(
                    points=points, transcription=transcription, language=[language],
                    illegibility=illegibility, orientation=orientation, word_tags=None
                )
                languages.add(language)
                word_idx += 1
            else:
                rects, words = polygon_to_rectangle(points, transcription)
                for idx in range(len(words)):
                    points = rects[idx]
                    points = np.array(points, dtype=np.float32).reshape(4, 2).tolist()
                    points = rearrange_points(points)
                    illegibility = words[idx] == '###'
                    orientation = 'Horizontal'
                    language = get_language_token(language)
                    words_info[word_idx] = dict(
                        points=points, transcription=words[idx],
                        language=[language], illegibility=illegibility,
                        orientation=orientation, word_tags=None
                    )
                    languages.add(language)
                    word_idx += 1
        return words_info, dict(languages=languages)

def main(args):
    dst_image_dir = osp.join(DST_DATASET_DIR, 'images')
    # dst_image_dir = None

    mlt_train = MLT17Dataset(osp.join(SRC_DATASET_DIR, 'raw/train_images'),
                             osp.join(SRC_DATASET_DIR, 'raw/training_gt'),
                             copy_images_to=dst_image_dir)
    mlt_merged = mlt_train
    if args.is_valid:
        mlt_valid = MLT17Dataset(osp.join(SRC_DATASET_DIR, 'raw/ch8_validation_images'),
                                osp.join(SRC_DATASET_DIR, 'raw/ch8_validation_gt'),
                                copy_images_to=dst_image_dir)
        mlt_merged = ConcatDataset([mlt_merged, mlt_valid])

    print(len(mlt_merged))
    anno = dict(images=dict())
    with tqdm(total=len(mlt_merged)) as pbar:
        for batch in DataLoader(mlt_merged, num_workers=NUM_WORKERS, collate_fn=lambda x: x):
            image_fname, sample_info = batch[0]
            anno['images'][image_fname] = sample_info
            pbar.update(1)

    ufo_dir = osp.join(DST_DATASET_DIR, 'ufo')
    maybe_mkdir(ufo_dir)
    with open(osp.join(ufo_dir, 'train.json'), 'w') as f:
        json.dump(anno, f, indent=4)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--is_valid', type=bool,
                        default=False)
    args = parser.parse_args()
    main(args)
