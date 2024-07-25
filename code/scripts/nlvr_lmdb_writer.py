# -*- coding: utf-8 -*-
# @time: 11/2/2023 1:06 PM
# @Author: 坤
# @file: nlvr_lmdb_writer.py
import lmdb
import threading

### !!! Change this to the directory
json_file = ['/zhaobai46a02/NLVR/annotations/nlvr_dev.json', '/zhaobai46a02/NLVR/annotations/nlvr_test.json',
             '/zhaobai46a02/NLVR/annotations/nlvr_train.json']
import json

annotations = []
for f in json_file:
    annotations += json.load(open(f, 'r'))

print(len(annotations))

lmdb_path = 'please input your lmdb save root'
image_root = 'please input your nlvr image root'
import lmdb

env = lmdb.open(lmdb_path, map_size=int(1e12))


def save_to_lmdb(image_path, idx):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    with env.begin(write=True) as txn:
        txn.put(idx.encode(), image_data)


import os

for i, ann in enumerate(annotations):
    images = ann['images']
    for path in images:
        image_path = os.path.join(image_root, path)
        save_to_lmdb(image_path, path)
        if i % 1000 == 0:
            print(f"{i}th record have been processed")

env.close()

print("transfer format end")
