# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, Sampler
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
from torch.utils.data import DistributedSampler
import lmdb
import math
import random
from operator import itemgetter
from typing import Iterator, List, Optional, Union

class cc3m_dataset_gain(Dataset):
    def __init__(self, transform, json_file, lmdb_root, max_words=30, caption_method='random', fraction = 1.):
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
        self.annotations = self.annotations[:int(fraction*len(self.annotations))]
        self.max_words = max_words
        self._create_env(lmdb_root)

        self.caption_method = caption_method

        self.gains = np.array([np.mean([float(x['gains'][0]),float(x['gains'][1])]) for x in self.annotations])

        upper = np.quantile(self.gains, 0.99)
        self.gains = self.gains/upper
        self.gains[self.gains>1.]=1.

        self.total_gain = sum(self.gains)

        if self.caption_method not in ['random', 'append', 'weight', 'first', 'best', 'nonlist']:
            raise ValueError('caption_method must be random or append or weight')

    def _create_env(self, path):
        self.env = lmdb.open(path, readonly=True, lock=False, readahead=False)

    def prune(self):
        r = np.random.rand(len(self.annotations))
        chosen = self.gains>=r
        idxlist = chosen.nonzero()[0].tolist()
        np.random.shuffle(idxlist)
        return idxlist

    def reverse_prune(self):
        r = np.random.rand(len(self.annotations))
        p = 1.-self.gains
        #p = [1. - x for x in self.gains]
        p[p<0.1]=0.1
        chosen = p>=r
        idxlist = chosen.nonzero()[0].tolist()
        np.random.shuffle(idxlist)
        return idxlist

    def sampler(self):
        return GainSampler(self)

    def twophasesampler(self):
        return GainSamplerV2(self)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        #image_path = os.path.basename(ann['image'])
        image_path = ann['image'][:-4]
        with self.env.begin() as txn:
            image_data = txn.get(image_path.encode())

        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # the image file is xxx.jpg, the label named by xxx.txt
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

        image = self.transform(image)

        return image, caption, image_path

class GainSampler():
    def __init__(self, cc3m_dataset_gain):
        self.cc3m_dataset = cc3m_dataset_gain
        self.seq = None
        self.seed = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed+=1
        self.seq = self.cc3m_dataset.prune()
        while len(self.seq)<len(self.cc3m_dataset):
            l = self.cc3m_dataset.prune()
            if len(l)>=len(self.cc3m_dataset)-len(self.seq):
                self.seq.extend(np.random.choice(l,len(self.cc3m_dataset)-len(self.seq),replace=False))
                break
            else:
                self.seq.extend(l)
        self.ite = iter(self.seq)

    def __next__(self):
        try:
            nxt = next(self.ite)
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self

class GainSamplerV2():
    def __init__(self, cc3m_dataset_gain):
        self.cc3m_dataset = cc3m_dataset_gain
        self.seq = None
        self.seed = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed+=1
        if self.seed%2==1:
            self.seq = self.cc3m_dataset.prune()
        else:
            self.seq = self.cc3m_dataset.reverse_prune()
        self.ite = iter(self.seq)

    def __next__(self):
        try:
            nxt = next(self.ite)
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self



class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler can change size during training.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
#         self.sampler.reset()
        self.dataset = DatasetFromSampler(self.sampler)
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))