import os
import json
import random

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image
import io
import lmdb
from data.utils import pre_caption

class nlvr_dataset(Dataset):
    def __init__(self, transform, image_root, ann_root, lmdb_root, split):
        '''
        image_root (string): Root directory of images 
        ann_root (string): directory to store the annotation file
        split (string): train, val or test
        '''
        urls = {'train':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_train.json',
                'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_dev.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_test.json'}
        filenames = {'train':'nlvr_train.json','val':'nlvr_dev.json','test':'nlvr_test.json'}
        
        download_url(urls[split],ann_root)
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        
        self.transform = transform
        self.image_root = image_root
        self.lmdb_root = lmdb_root
        self._create_env()

    def _create_env(self):
        self.env = lmdb.open(self.lmdb_root, readonly=True, lock=False, readahead=False)
        
    def __len__(self):
        return len(self.annotation)
    

    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image0_path = os.path.join(self.image_root,ann['images'][0])
        with self.env.begin() as txn:
            image_data = txn.get(image0_path.encode())

        image0 = Image.open(io.BytesIO(image_data)).convert('RGB')
        image0 = self.transform(image0)   
        
        image1_path = os.path.join(self.image_root,ann['images'][1])
        with self.env.begin() as txn:
            image_data = txn.get(image1_path.encode())

        image1 = Image.open(io.BytesIO(image_data)).convert('RGB')
        image1 = self.transform(image1)          

        sentence = pre_caption(ann['sentence'], 40)
        
        if ann['label']=='True':
            label = 1
        else:
            label = 0
            
        words = sentence.split(' ')
        
        if 'left' not in words and 'right' not in words:
            if random.random()<0.5:
                return image0, image1, sentence, label
            else:
                return image1, image0, sentence, label
        else:
            if random.random()<0.5:
                return image0, image1, sentence, label
            else:
                new_words = []
                for word in words:
                    if word=='left':
                        new_words.append('right')
                    elif word=='right':
                        new_words.append('left')        
                    else:
                        new_words.append(word)                    
                        
                sentence = ' '.join(new_words)
                return image1, image0, sentence, label
            
            
        