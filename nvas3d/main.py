#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import yaml
import shutil
import logging
import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from nvas3d.train.trainer import Trainer
from nvas3d.data_loader.data_loader import SSAVDataLoader
from nvas3d.model.model import NVASNet


def setup_distributed_training(local_rank):
    world_size = torch.cuda.device_count()
    dist.init_process_group(backend='nccl',
                            init_method='tcp://localhost:12355',
                            rank=local_rank,
                            world_size=world_size)
    device = torch.device(f'cuda:{local_rank}')
    return device


def main(local_rank, args):
    is_ddp = args.gpu is None
    device = setup_distributed_training(local_rank) if is_ddp else torch.device(f'cuda:{args.gpu}')

    if not is_ddp:
        torch.cuda.set_device(args.gpu)

    # Load and parse config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create a directory and copy config
    save_dir = os.path.join(config['save_dir'], f'{args.exp}')
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(args.config, f'{save_dir}/config.yaml')

    # Initialize DataLoader
    data_loader = SSAVDataLoader(config['use_visual'], config['use_deconv'], is_ddp, **config['data_loader'])

    # Initialize and deploy model to device
    model = NVASNet(config['data_loader']['num_receivers'], config['use_visual'])
    model = model.to(device)
    if is_ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # Train the model
    trainer = Trainer(model, data_loader, device, save_dir, config['use_deconv'], config['training'])
    trainer.train()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    torch.manual_seed(42)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='./nvas3d/config/default_config.yaml',
                        help='Path to the configuration file.')

    parser.add_argument('--exp',
                        type=str,
                        default='default_exp',
                        help='Experiment name')

    parser.add_argument('--gpu',
                        type=int,
                        default=None,
                        help='GPU ID to use')

    args = parser.parse_args()

    if args.gpu is not None:
        main(0, args)  # Single GPU mode
    else:
        mp.spawn(main, args=(args,), nprocs=torch.cuda.device_count(), join=True)
