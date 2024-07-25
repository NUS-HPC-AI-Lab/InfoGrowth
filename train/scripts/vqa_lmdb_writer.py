# -*- coding: utf-8 -*-
# @time: 11/2/2023 1:14 PM
# @Author: 坤
# @file: vqa_lmdb_writer.py
import lmdb
import threading

# change this to the directory
json_file = ['/zhaobai46a02/VQAv2/annotations/vqa_test.json']
import json
import os

annotations = []
for f in json_file:
    annotations += json.load(open(f, 'r'))

print(len(annotations))
image_path_list = set()

lmdb_path = '/zhaobai46a02/VQAv2/lmdb'
vqa_image_root = '/zhaobai46a02/VQAv2'
vg_image_root = '/zhaobai/VisualGenome'
import lmdb

for ann in annotations:
    image_path = ann['image']
    if ann['dataset'] == 'vqa':
        idx = os.path.join('vqa', image_path)
    elif ann['dataset'] == 'vg':
        idx = os.path.join('vg', image_path)
    image_path_list.add(idx)
print("image to save length: ", len(image_path_list))

env = lmdb.open(lmdb_path, map_size=int(1e12))


def save_to_lmdb(image_path, idx):
    with env.begin(write=False) as txn:
        if txn.get(idx.encode()) is not None:
            return
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        with env.begin(write=True) as txn:
            txn.put(idx.encode(), image_data)
    except:
        print("no such file {}".format(image_path))

for i, image_path in enumerate(image_path_list):
    # if is vg path
    if image_path.startswith('vg'):
        image_path = image_path[3:]
        read_path = os.path.join(vg_image_root, image_path)
    elif image_path.startswith('vqa'):
        image_path = image_path[4:]
        read_path = os.path.join(vqa_image_root, image_path)

    save_to_lmdb(read_path, image_path)

    if i % 1000 == 0:
        print("save {} images".format(i))

env.close()

print("transfer format end")
