# -*- coding: utf-8 -*-
# @time: 10/17/2023 1:26 AM
# @Author: 坤
# @file: finetune_retrieval.py

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/home/kun/data/CC12M/images')
parser.add_argument('--pretrained', type=str, default='')
parser.add_argument('--output_dir', type=str, default='output/finetune_retrieval')

args = parser.parse_args()

# first run train_retrieval.py
import os
coco_lmdb_root = os.path.join(args.data_root, 'MSCOCO', 'lmdb')
pretrained = args.pretrained
coco_output_dir = os.path.join(args.output_dir, 'coco')
coco_config = 'configs/retrieval_coco.yaml'
coco_ann_root = os.path.join(args.data_root, 'MSCOCO', 'annotations')

print("starting coco zero-shot")
os.system('python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py --config {} --pretrained {} --output_dir {} --ann_root {} --lmdb_root {} --evaluate > {}/coco_retrieval_zero_shot.log'
    .format(coco_config, pretrained, coco_output_dir, coco_ann_root, coco_lmdb_root, args.output_dir))

print("starting coco finetune")
os.system('python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py --config {} --pretrained {} --output_dir {} --ann_root {} --lmdb_root {} > {}/coco_retrieval_finetune.log'.format(
    coco_config, pretrained, coco_output_dir, coco_ann_root, coco_lmdb_root, args.output_dir))

flickr_lmdb_root = os.path.join(args.data_root, 'Flickr30k', 'lmdb')
flickr_output_dir = os.path.join(args.output_dir, 'flickr')
flickr_config = 'configs/retrieval_flickr.yaml'
flickr_ann_root = os.path.join(args.data_root, 'Flickr30k', 'annotations')
flickr_pretrained = os.path.join(coco_output_dir, 'checkpoint_best.pth')
print("starting flickr zero shot")
os.system('python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py --config {} --pretrained {} --output_dir {} --ann_root {} --lmdb_root {} --evaluate > {}/flickr_retrieval_zero_shot.log'.format(
    flickr_config, flickr_pretrained, flickr_output_dir, flickr_ann_root, flickr_lmdb_root, args.output_dir))

print("starting flickr finetune")
os.system('python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py --config {} --pretrained {} --output_dir {} --ann_root {} --lmdb_root {} > {}/flickr_retrieval_finetune.log'.format(
    flickr_config, pretrained, flickr_output_dir, flickr_ann_root, flickr_lmdb_root, args.output_dir))

print("Done!")