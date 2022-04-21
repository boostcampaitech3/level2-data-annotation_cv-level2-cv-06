import os
import os.path as osp
import time
import math
import wandb
from datetime import timedelta
from argparse import ArgumentParser

###추가
from detect import get_bboxes
from deteval import calc_deteval_metrics
import numpy as np

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/Combine_Data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--wandb_project', type=str, default="data_anno")
    parser.add_argument('--wandb_interval', type=int, default=5)
    parser.add_argument('--wandb_num_examples', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, wandb_project, wandb_interval,
                wandb_num_examples):
    exp_name = '_'.join(map(str, [image_size, input_size, batch_size, learning_rate, max_epoch,"AdamW",'Combine_data']))
    
    wandb.init(
        project=wandb_project,
        name="EAST_" + exp_name,
        entity="jhwan96")
    
    dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0)
    wandb.watch(model, log='all')


    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        gt_bboxes, pred_bboxes, trans = [], [], []
        with tqdm(total=num_batches) as pbar:
            for step, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(train_loader):
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                ####
                orig_sizes = []
                for image in img:
                    orig_sizes.append(image.shape[1:3])
                gt_bbox = []
                pred_bbox = []
                tran = []
                with torch.no_grad():
                    pred_score_map, pred_geo_map = model.forward(img.to(device))
                
                
                for gt_score, gt_geo, pred_score, pred_geo, orig_size in zip(gt_score_map.cpu().numpy(), gt_geo_map.cpu().numpy(), pred_score_map.cpu().numpy(), pred_geo_map.cpu().numpy(), orig_sizes):
                    gt_bbox_angle = get_bboxes(gt_score, gt_geo)
                    pred_bbox_angle = get_bboxes(pred_score, pred_geo)
                    if gt_bbox_angle is None:
                        gt_bbox_angle = np.zeros((0, 4, 2), dtype=np.float32)
                        tran_angle = []
                    else:
                        gt_bbox_angle = gt_bbox_angle[:, :8].reshape(-1, 4, 2)
                        gt_bbox_angle *= max(orig_size) / input_size
                        tran_angle = ['null' for _ in range(gt_bbox_angle.shape[0])]
                    if pred_bbox_angle is None:
                        pred_bbox_angle = np.zeros((0, 4, 2), dtype=np.float32)
                    else:
                        pred_bbox_angle = pred_bbox_angle[:, :8].reshape(-1, 4, 2)
                        pred_bbox_angle *= max(orig_size) / input_size
                        
                    tran.append(tran_angle)
                    gt_bbox.append(gt_bbox_angle)
                    pred_bbox.append(pred_bbox_angle)
                    

                gt_bboxes.extend(gt_bbox)
                pred_bboxes.extend(pred_bbox)
                trans.extend(tran)

                ####

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }

                if step % wandb_interval == 0:
                    wandb.log({
                        "train/loss": loss_val,
                        "train/Cls loss": extra_info['cls_loss'],
                        "train/Angle loss": extra_info['angle_loss'],
                        "train/IoU loss": extra_info['iou_loss'],
                        "train/Learning Rate": optimizer.param_groups[0]['lr'],
                        "epoch": epoch+1
                    })

                pbar.set_postfix(val_dict)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        ####
        img_len = len(dataset)
        pred_bboxes_dict, gt_bboxes_dict, trans_dict = dict(), dict(), dict()
        for img_num in range(img_len):
            pred_bboxes_dict[f'img_{img_num}'] = pred_bboxes[img_num]
            gt_bboxes_dict[f'img_{img_num}'] = gt_bboxes[img_num]
            trans_dict[f'img_{img_num}'] = trans[img_num]
        
        deteval_dict = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, trans_dict)
        metric_dict = deteval_dict['total']
        precision = metric_dict['precision']
        recall = metric_dict['recall']
        hmean = metric_dict['hmean']
        print(f"precision : {precision}, recall : {recall}, hmean : {hmean}")
        ####


        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
