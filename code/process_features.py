# -*- coding: utf-8 -*-
# @time: 10/13/2023 7:59 PM
# @Author: 坤
# @file: compression.py
import argparse
import os
import pdb

try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip_pretrain import blip_pretrain
from models.blip_sim import  blip_sim
import utils
from utils import warmup_lr_schedule, step_lr_schedule
from timm.data.loader import PrefetchLoader
from data import create_dataset, create_sampler, create_loader
import timm
from torch.utils.tensorboard import SummaryWriter
from compression import *

def image_text_graph(data_loader, device, config):
    # load image feature and caption feature, with image index and caption index
    if config['cal_feat']:
        print("Creating model")
        model = blip_sim(image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                         vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])

        

        if config['checkpoint']:
            checkpoint = torch.load(config['checkpoint'], map_location='cpu')
            state_dict = checkpoint['model']
            model.load_state_dict(state_dict)
            print('resume checkpoint from %s' % config['checkpoint'])

        model = model.to(device)

        # if args.distributed:
        #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        image_feature_array = []
        text_feature_array = []
        data_index_list = []
        # get the feature of the whole dataset, image feature and text feature
        for batch_idx, (images, caption, image_path) in enumerate(data_loader):
            images = images.to(device)

            with torch.no_grad():
                image_feat = model.get_image_feature(images)
                text_feat = model.get_text_feature(caption, 'cuda')

            
            image_feature_array.append(image_feat.cpu())
            text_feature_array.append(text_feat.cpu())


            for i, path in enumerate(image_path):
                data_index_list.append({'image_path': path, 'text': caption[i]})
            if batch_idx % 5 == 0:
                print('sampled %d batches' % batch_idx)
                print("image_feature_array shape: ", len(image_feature_array) * image_feature_array[0].shape[0])
                print("text_feature_array shape: ", len(text_feature_array) * text_feature_array[0].shape[0])


        print("processing finished")
        print("image_feature_array shape: ", len(image_feature_array) * image_feature_array[0].shape[0])
        print("text_feature_array shape: ", len(text_feature_array) * text_feature_array[0].shape[0])


        image_feature_array = torch.cat(image_feature_array, dim=0)
        text_feature_array = torch.cat(text_feature_array, dim=0)

        # save the feature
        save_obj = {
            'image_feature_array': image_feature_array,
            'text_feature_array': text_feature_array,
            'image_path': data_index_list
        }
        torch.save(save_obj, os.path.join(config['save_root'], config['feature_save_file']))
        print("feature saved to %s" % os.path.join(config['save_root'], config['feature_save_file']))

class PrefetchLoaderWrapper(PrefetchLoader):
    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target, next_path in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                # next_target = next_target.cuda(non_blocking=True)

            if not first:
                yield input, target, path
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target
            path = next_path

        yield input, target, path


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    datasets = [create_dataset(config['dataset'], config)]
    print('number of training samples: %d' % len(datasets[0]))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

#     samplers = create_sampler(datasets, [False], num_tasks, global_rank)
    samplers = create_sampler(datasets, [True], num_tasks, global_rank)

    data_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']], num_workers=[4], is_trains=[False],
                  collate_fns=[None])[0]
    # data_loader = PrefetchLoaderWrapper(data_loader)

    image_text_graph(data_loader, 'cuda', config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/feature_processing.yaml')
    parser.add_argument('--output_dir', default='/data/common/cc3m/blip_features')
    parser.add_argument('--checkpoint', default='/data/personal/nus-qzh/blip-base/blip_149_model_base.pth')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)