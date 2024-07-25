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
import utils
from utils import warmup_lr_schedule, step_lr_schedule, cosine_lr_schedule
from timm.data.loader import PrefetchLoader
from data import create_dataset, create_sampler, create_loader
import timm
from torch.utils.tensorboard import SummaryWriter
from data.cc3m_dataset_with_gains import DistributedSamplerWrapper

class PrefetchLoaderWrapper(PrefetchLoader):
    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target, _ in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                # next_target = next_target.cuda(non_blocking=True)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

def train(model, data_loader, optimizer, epoch, device, config, writer=None):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    # if config['laion_path']:
    #     data_loader.dataset.reload_laion(epoch)


    data_loader.sampler.set_epoch(epoch)

    for i, (image, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if epoch==0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])

        optimizer.zero_grad()


        # image = image.to(device,non_blocking=True)

        # ramp up alpha in the first 2 epochs
        alpha = config['alpha']*min(1,(epoch*len(data_loader)+i)/(2*len(data_loader)))

        loss_ita, loss_itm, loss_lm = model(image, caption, alpha = alpha)
        loss = loss_ita + loss_itm + loss_lm

        loss.backward()
        optimizer.step()

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_lm=loss_lm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # log in tensorboard
        # if writer is not None and utils.is_main_process():
        #     writer.add_scalar('loss_ita', loss_ita.item(), epoch*len(data_loader)+i)
        #     writer.add_scalar('loss_itm', loss_itm.item(), epoch*len(data_loader)+i)
        #     writer.add_scalar('loss_lm', loss_lm.item(), epoch*len(data_loader)+i)
        #     writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch*len(data_loader)+i)
        dist.barrier()


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    datasets = [create_dataset(config['dataset'], config, min_scale=0.2)]
    print('number of training samples: %d'%len(datasets[0]))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if args.sampler=='proptional':
        samplers = [DistributedSamplerWrapper(datasets[0].sampler(), num_replicas=num_tasks,
                                                  rank=global_rank, shuffle=False)]
    else:
        samplers = [DistributedSamplerWrapper(datasets[0].twophasesampler(), num_replicas=num_tasks,
                                                  rank=global_rank, shuffle=False)]

    data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[1], is_trains=[True], collate_fns=[None])[0]

    data_loader = PrefetchLoaderWrapper(data_loader)

    #### Model ####
    print("Creating model")
    model = blip_pretrain(image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                            vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])

    model = model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    start_epoch = 0
    cheak_points = False
    for i in range(19):
        files = args.output_dir + '/checkpoint_%02d.pth'%i
        if os.path.isfile(files):
            cheak_points = files
    if cheak_points:
        checkpoint = torch.load(cheak_points, map_location='cpu') 
        state_dict = checkpoint['model']    
        model.load_state_dict(state_dict)    
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']+1                
        print('resume checkpoint from output %s'%checkpoint)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']+1
        print('resume checkpoint from %s'%args.checkpoint)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    writer = None
    for epoch in range(start_epoch, config['max_epoch']):
        if config['decay_method'] == 'step':
            step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])
        elif config['decay_method'] == 'cosine':
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
        else:
            raise NotImplementedError

        train_stats = train(model, data_loader, optimizer, epoch, device, config, writer=writer)
        if utils.is_main_process():
            if epoch in range(20) or epoch == config['max_epoch']-1:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch,
                            }
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))
                if epoch>0:
                    os.remove(os.path.join(args.output_dir, 'checkpoint_%02d.pth'%(epoch-1)))
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/pretrain_cc.yaml')
    parser.add_argument('--output_dir', default='output/pretrain_cc_merge_caption1')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--lmdb_root', type=str, help='dataset lmdb root')
    parser.add_argument('--ann_root', type=str, help='dataset json path')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--sampler', type=str, default='proptional')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    if args.lmdb_root:
        config['lmdb_root'] = args.lmdb_root
    if args.ann_root:
        config['ann_root'] = args.ann_root
    if args.batch_size:
        config['batch_size'] = args.batch_size

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)