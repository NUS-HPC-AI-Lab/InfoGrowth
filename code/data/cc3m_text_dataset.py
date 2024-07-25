# -*- coding: utf-8 -*-
# @time: 10/6/2023 6:41 PM
# @Author: 坤
# @file: cc3m_dataset.py
from torch.utils.data import Dataset
import glob, os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
Image.MAX_IMAGE_PIXELS = None
from data.utils import pre_caption
import json
from data.utils import pre_caption
import lmdb
import random

class cc3m_text_dataset(Dataset):
    def __init__(self, transform, json_file, lmdb_root, max_words=30, caption_method='random'):
        self.transform = transform

        # "the file structure of cc3m:"
        # "1\000001.jpg, 000002.jpg, ..., 001000.jpg"
        # "2\001001.jpg, 001002.jpg, ..., 002000.jpg"
        # "..."
        if isinstance(json_file, list):
            self.annotations = []
            for f in json_file:
                self.annotations += json.load(open(f, 'r'))
        else:
            self.annotations = json.load(open(json_file, 'r'))
        self.max_words = max_words

        self.caption_method = caption_method

        if self.caption_method not in ['random', 'append', 'weight', 'first', 'best', 'nonlist']:
            raise ValueError('caption_method must be random or append or weight')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        image_path = os.path.basename(ann['image'])
        label = ann['caption']

        if isinstance(label, list):
            if self.caption_method == 'random':
                label = random.choice(label)
            elif self.caption_method == 'append':
                label = ' '.join(label)
            elif self.caption_method == 'weight':
                weight = np.array(ann['raw_score'])
                weight = weight / weight.sum()
                # normalize the weight
                label = random.choices(label, weights=weight)[0]
            elif self.caption_method == 'first':
                label = label[0]
            elif self.caption_method == 'best':
                label = label[np.argmax(ann['raw_score'])]

        if isinstance(label, str):
            label = label.strip().replace('\n', ' ')

        if isinstance(label, list):
            for i in range(len(label)):
                label[i] = label[i].strip().replace('\n', ' ')
            caption = [pre_caption(l, self.max_words) for l in label]

        else:
            caption = pre_caption(label, self.max_words)

        return caption, image_path

