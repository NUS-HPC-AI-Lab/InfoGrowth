# -*- coding: utf-8 -*-
# @time: 10/16/2023 10:08 PM
# @Author: 坤
# @file: cc3m_lmdb_writer.py
import lmdb
import argparse
import os
import glob
import json

parser = argparse.ArgumentParser(description='CC3M LMDB Writer')
parser.add_argument('--image_root', type=str, default='data/cc3m', help='image root')

args = parser.parse_args()

train_root = os.path.join(args.image_root, 'cc3m_train')
val_root = os.path.join(args.image_root, 'cc3m_val')

train_lmdb_root = os.path.join(args.image_root, 'lmdb_train')
val_lmdb_root = os.path.join(args.image_root, 'lmdb_val')

train_json_file = os.path.join(args.image_root, 'cc3m_train.json')
val_json_file = os.path.join(args.image_root, 'cc3m_val.json')



# process train data
print("process train data")
train_annotations = []
lmdb_train_env = lmdb.open(train_lmdb_root, map_size=int(1e12))
num = 0
for i, path in enumerate(glob.glob(os.path.join(train_root, '*.jpg'))):
    caption = json.load(open(path.replace('.jpg', '.json')))['caption']
    file_name = os.path.basename(path)

    ann = {'image': file_name, 'caption': caption}


    train_annotations.append(ann)

    # write to lmdb
    with open(path, 'rb') as f:
        image_data = f.read()
    with lmdb_train_env.begin(write=True) as txn:
        txn.put(file_name.encode(), image_data)
    num += 1
    if i % 5000 == 0:
        print(f"process {i} images")
lmdb_train_env.close()
print(f"train data length: {num}")
print("json save at {}".format(train_json_file), "lmdb save at {}".format(train_lmdb_root))

# process val data
print("process val data")
val_annotations = []
num = 0
lmdb_val_env = lmdb.open(val_lmdb_root, map_size=int(1e12))
for i, path in enumerate(glob.glob(os.path.join(val_root, '*.jpg'))):
    caption = json.load(open(path.replace('.jpg', '.json')))['caption']
    file_name = os.path.basename(path)
    ann = {'image': file_name, 'caption': caption}


    val_annotations.append(ann)

    # write to lmdb
    with open(path, 'rb') as f:
        image_data = f.read()
    with lmdb_val_env.begin(write=True) as txn:
        txn.put(file_name.encode(), image_data)
    num += 1
    if i % 5000 == 0:
        print(f"process {i} images")

lmdb_val_env.close()
print("val data length: {}".format(num))
print("json save at {}".format(val_json_file), "lmdb save at {}".format(val_lmdb_root))