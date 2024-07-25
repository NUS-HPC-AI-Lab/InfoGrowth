# -*- coding: utf-8 -*-
# @time: 11/8/2023 8:15 PM
# @Author: 坤
# @file: nocaps_lmdb_writer.py

# -*- coding: utf-8 -*-
# @time: 11/2/2023 1:06 PM
# @Author: 坤
# @file: nlvr_lmdb_writer.py

### !!! Change this to the directory
json_file = ['/zhaobai46a02/nocaps/Nocaps/annotations/nocaps_test.json', '/zhaobai46a02/nocaps/Nocaps/annotations/nocaps_val.json']
import json

annotations = []
for f in json_file:
    annotations += json.load(open(f, 'r'))

print(len(annotations))

lmdb_path = '/zhaobai46a02/nocaps/lmdb'
image_root = '/zhaobai46a02/nocaps/'
import lmdb

env = lmdb.open(lmdb_path, map_size=int(1e12))


def save_to_lmdb(image_path, idx):

    # search if exist
    with env.begin(write=False) as txn:
        if txn.get(idx.encode()) is not None:
            return

    with open(image_path, 'rb') as f:
        image_data = f.read()
    with env.begin(write=True) as txn:
        txn.put(idx.encode(), image_data)


import os

for i, ann in enumerate(annotations):
    path = ann['image']
    image_path = os.path.join(image_root, path)
    save_to_lmdb(image_path, path)
    if i % 1000 == 0:
        print(f"{i}th record have been processed")

env.close()

print("transfer format end")