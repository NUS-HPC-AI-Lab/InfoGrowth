# -*- coding: utf-8 -*-
# @time: 10/17/2023 1:03 AM
# @Author: 坤
# @file: retrieval_lmdb_writer.py

import argparse
import os

parser = argparse.ArgumentParser(description='retrieval_lmdb_writer')
parser.add_argument('--data_root', default='/home/kun/data/cc3m', type=str)

args = parser.parse_args()

coco_annotation_root = os.path.join(args.data_root, 'MSCOCO/annotations')
flickr30_annotation_root = os.path.join(args.data_root, 'Flickr30k/annotations')

coco_lmdb_root = os.path.join(args.data_root, 'MSCOCO/lmdb')
flickr30_lmdb_root = os.path.join(args.data_root, 'Flickr30k/lmdb')

# process coco file
import glob
import json
coco_annotations = []
for file in glob.glob(os.path.join(coco_annotation_root, '*.json')):
    data = json.load(open(file, 'r'))
    coco_annotations += data

print("coco_annotations: ", len(coco_annotations))

# write in lmdb
import lmdb
coco_lmdb_env = lmdb.open(coco_lmdb_root, map_size=int(1e12))
num = 0
for i, ann in enumerate(coco_annotations):
    path = os.path.join(args.data_root, 'MSCOCO', ann['image'])
    with open(path, 'rb') as f:
        image_data = f.read()
    with coco_lmdb_env.begin(write=True) as txn:
        txn.put(ann['image'].encode(), image_data)
    num += 1
    if i % 5000 == 0:
        print(f"process {i} images")
coco_lmdb_env.close()
print(f"coco data length: {num}")

# process flickr30 file
flickr30_annotations = []
for file in glob.glob(os.path.join(flickr30_annotation_root, '*.json')):
    data = json.load(open(file, 'r'))
    flickr30_annotations += data
print("flickr30_annotations: ", len(flickr30_annotations))


# write in lmdb
flickr30_lmdb_env = lmdb.open(flickr30_lmdb_root, map_size=int(1e12))
num = 0
for i, ann in enumerate(flickr30_annotations):
    path = os.path.join(args.data_root, 'Flickr30k', ann['image'])
    with open(path, 'rb') as f:
        image_data = f.read()
    with flickr30_lmdb_env.begin(write=True) as txn:
        txn.put(ann['image'].encode(), image_data)
    num += 1
    if i % 5000 == 0:
        print(f"process {i} images")
flickr30_lmdb_env.close()
print(f"flickr30 data length: {num}")
