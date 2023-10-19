#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import glob
import time
import logging
import typing as T
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from nvas3d.utils.audio_utils import spec2wav, compute_metrics
from nvas3d.utils.utils import source_class_map, get_key_from_value
from nvas3d.utils.plot_utils import plot_debug


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        data_loader,
        device: torch.device,
        save_dir: str,
        use_deconv: bool,
        config: T.Dict[str, T.Any],
        save_audio: bool = False
    ) -> None:
        """
        Initializes the Trainer class.

        Args:
        - model:        Model to be trained.
        - data_loader:  Data loader supplying training data.
        - device:       Device (CPU or GPU) to be used for training.
        - save_dir:     Directory to save the model.
        - use_deconv:   Flag to use deconvolution.
        - config:       Configuration with training parameters.
        """

        # Args
        self.model = model.to(device, dtype=torch.float)
        self.data_loader = data_loader
        self.device = device
        self.save_dir = save_dir
        self.use_deconv = use_deconv
        self.save_audio = save_audio

        # Optimzer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # Loss
        self.regressor_criterion = nn.MSELoss().to(device=self.device)
        self.bce_criterion = nn.BCEWithLogitsLoss().to(device=self.device)

        # Training configuration
        self.epochs = config['epochs']
        self.resume = config['resume']
        self.detect_loss_weight = config['detect_loss_weight']
        self.save_checkpoint_interval = config['save_checkpoint_interval']

        # Tensorboard
        tb_dir = f'{self.save_dir}/tb'
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_dir)

        # Resume
        if self.resume:
            print(f'Resume from {config["checkpoint_dir"]}...')
            latest_checkpoint = self.find_latest_checkpoint(config['checkpoint_dir'])
            if latest_checkpoint is not None:
                start_epoch = self.load_checkpoint(latest_checkpoint) + 1
            else:
                start_epoch = 0
        else:
            start_epoch = 0
        self.start_epoch = start_epoch

    def train(self):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, pow(0.1, 1 / self.epochs))

        since = time.time()

        for epoch in range(self.start_epoch, self.epochs + 1):
            logging.info('-' * 10)
            logging.info('Epoch {}/{}'.format(epoch, self.epochs))

            # Set DistributedSampler for each epoch if model is DDP
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) and self.model.training:
                self.data_loader.train_sampler.set_epoch(epoch)

            # Initialize variables to store metrics
            running_loss = defaultdict(float)
            running_metrics = defaultdict(float)
            num_data_point = defaultdict(int)
            count_metrics = defaultdict(int)

            for train_mode in ['Train', 'Val']:
                # Skip validation if not saving checkpoint this epoch
                if train_mode == 'Val' and epoch % self.save_checkpoint_interval != 0:
                    continue

                data_loader = (self.data_loader.get_train_data() if train_mode == 'Train'
                               else self.data_loader.get_val_data())

                # Set mode
                if train_mode == 'Train':
                    self.model.train()
                else:
                    self.model.eval()

                with torch.set_grad_enabled(train_mode == 'Train'):
                    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
                    for data, room, id in pbar:
                        # Preprocess
                        for key, value in data.items():
                            data[key] = value.to(device=self.device, dtype=torch.float)

                        # Set mask
                        positive_mask = (data['is_source'] > 0.5).reshape(-1)
                        nonzero_mask = torch.any(data['source1_audio'], dim=1)
                        nan_mask = torch.any(torch.isnan(data['source1_audio']), dim=1)
                        valid_mask = nonzero_mask & ~nan_mask
                        valid_positive_mask = positive_mask & valid_mask
                        if nan_mask.any():
                            nan_room = [x for x, m in zip(room, nan_mask) if m]
                            nan_id = [x for x, m in zip(id, nan_mask) if m]
                            print(f'{nan_room}, {nan_id}: invalid audio')
                            data['input_stft'][nan_mask] = 0  # will ignore in the loss

                        # Parse data
                        source1_stft = data['source1_stft']  # [n_batch, 2, F, N]
                        input_stft = data['input_stft']  # [n_batch, n_receivers*2, F, N]

                        source1_stft_nondc = source1_stft[:, :, 1:, :]

                        # Zero the parameter gradients
                        self.optimizer.zero_grad()

                        # Forward
                        output = self.model(data)
                        pred_stft = output['pred_stft']  # [n_batch, n_receivers*2, F, N]

                        # Compute detection loss on valid mask
                        detect_loss = self.bce_criterion(output['source_detection'][valid_mask], data['is_source'][valid_mask])

                        # Compute magnitude loss on valid positive mask, if any
                        mag_loss = (self.regressor_criterion(pred_stft[valid_positive_mask], source1_stft_nondc[valid_positive_mask])
                                    if valid_positive_mask.any() else torch.tensor(0.0, device=self.device))

                        # Combine losses
                        loss = mag_loss + detect_loss * self.detect_loss_weight
                        assert not torch.isnan(loss), 'Loss is NaN'

                        # Backward
                        if train_mode == 'Train':
                            loss.backward()
                            self.optimizer.step()

                        # Save metrics
                        if epoch % self.save_checkpoint_interval == 0 and valid_positive_mask.any():
                            save_idx = valid_positive_mask.nonzero(as_tuple=True)[0][0]
                            room = room[save_idx]
                            id_value = int(id[save_idx])

                            # Prepare audio data
                            full_pred_stft_save = input_stft[save_idx].clone().permute(1, 2, 0)[..., :2]
                            pred_stft_save = pred_stft[save_idx].permute(1, 2, 0)
                            full_pred_stft_save[1:, :, :] = pred_stft_save  # keep dc from input
                            pred_audio = spec2wav(full_pred_stft_save)
                            source1_audio_save = data['source1_audio'][save_idx]

                            # Save debug data
                            if self.save_audio:
                                self.save_audio_data(save_idx, pred_audio, source1_audio_save, data, train_mode, room, id_value, epoch)

                            # Compute metrics for dry-sound estimation
                            dry_psnr, dry_sdr = compute_metrics(pred_audio, source1_audio_save)

                            # Compute metrics for detection
                            predicted = (output['source_detection'] > 0.5).int()
                            accuracy_true = (predicted[data['is_source'] == 1] == 1).float().mean().item()
                            accuracy_false = (predicted[data['is_source'] == 0] == 0).float().mean().item()
                            detection_accuracy = (predicted == data['is_source']).float().mean().item()

                            metrics_dict = {
                                'dry_psnr': dry_psnr,
                                'dry_sdr': dry_sdr,
                                'detection_accuracy': detection_accuracy,
                                'accuracy_true': accuracy_true,
                                'accuracy_false': accuracy_false
                            }

                            # Metric reduction for DDP
                            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                                for metric, value in metrics_dict.items():
                                    value_tensor = torch.tensor(value).to(self.device)
                                    dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
                                    running_metrics[f'{train_mode}/{metric}'] += value_tensor.item()
                                    count_metrics[f'{train_mode}/{metric}'] += dist.get_world_size()
                            else:
                                for metric, value in metrics_dict.items():
                                    running_metrics[f'{train_mode}/{metric}'] += value
                                    count_metrics[f'{train_mode}/{metric}'] += 1

                        # Loss reduction for DDP
                        loss_dict = {
                            'total_loss': loss,
                            'mag_loss': mag_loss,
                            'detect_loss': detect_loss
                        }
                        B = input_stft.shape[0]

                        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                            for loss_type, loss_value in loss_dict.items():
                                loss_tensor = torch.tensor(loss_value).to(self.device)
                                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                                normalized_loss = loss_tensor.item() / dist.get_world_size()
                                running_loss[f'{train_mode}/{loss_type}'] += normalized_loss
                        else:
                            for loss_type, loss_value in loss_dict.items():
                                current_loss = loss_dict[loss_type].item() * B
                                running_loss[f'{train_mode}/{loss_type}'] += current_loss
                        num_data_point[train_mode] += B

            # Log, save checkpoints, write tensorboard
            should_log = not isinstance(self.model, torch.nn.parallel.DistributedDataParallel) or (dist.get_rank() == 0)
            if should_log:
                # Log and save checkpoints
                self.log_and_save(running_loss, running_metrics, epoch)

                # Write tensorboard
                for key, value in running_loss.items():
                    self.writer.add_scalar(key, value, epoch)
                for key, value in running_metrics.items():
                    self.writer.add_scalar(key, value, epoch)

            scheduler.step()

        time_elapsed = time.time() - since
        logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        print('Training finished.')

    def save_checkpoint(self, epoch):
        """
        Save checkpoint.
        """

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }

        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }

        filename = f'checkpoints/checkpoint_{epoch}.pt'
        os.makedirs(os.path.join(self.save_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.save_dir, filename))

    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint.
        """

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    def find_latest_checkpoint(self, checkpoint_dir):
        """
        Find latest checkpoint.
        """

        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '**', 'checkpoint_*.pt'), recursive=True)
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            return latest_checkpoint
        else:
            return None

    def log_and_save(self, running_loss, running_metrics, epoch):
        """
        Log values to Tensorboard and save model checkpoint.
        """

        # Logging to Tensorboard
        for key, value in running_loss.items():
            self.writer.add_scalar(key, value, epoch)
        for key, value in running_metrics.items():
            self.writer.add_scalar(key, value, epoch)

        # Save checkpoint
        if epoch % self.save_checkpoint_interval == 0:
            self.save_checkpoint(epoch)

    def save_audio_data(self, save_idx, pred_audio, source1_audio_save, data, train_mode, room, id_value, epoch, sample_rate=48000, save_normalized=True):
        """
        Function to save audio data.
        """

        # Set directory
        source1_class_save = get_key_from_value(source_class_map, data['source1_class'][save_idx])
        source2_class_save = get_key_from_value(source_class_map, data['source2_class'][save_idx])
        debug_dir = f'{self.save_dir}/{train_mode}_audio/{source1_class_save}-{source2_class_save}'

        # Prepare audio
        input_audio_mean = data['input_audio'][save_idx].mean(dim=0)
        audio_name_list = ['pred', 'input_mean', 'pred_gt', 'input_0', 'receiver_0']
        audio_data_list = [pred_audio, input_audio_mean, source1_audio_save, data['input_audio'][save_idx][0], data['receiver_audio'][save_idx][0]]

        # Save
        os.makedirs(f'{debug_dir}/{room}/{id_value}', exist_ok=True)
        for audio_name, audio_data in zip(audio_name_list, audio_data_list):
            filename_val = f'{debug_dir}/{room}/{id_value}/{epoch}_{audio_name}'
            plot_debug(filename_val, audio_data.reshape(1, -1).detach().cpu(),
                       sample_rate=sample_rate, save_normalized=save_normalized, reference=source1_audio_save)

        # Save RGB image if available
        if 'rgb' in data.keys():
            filename_val = f'{debug_dir}/{room}/{id_value}/{epoch}_rgb.png'
            plt.imsave(filename_val, data['rgb'][save_idx].permute(1, 2, 0).detach().cpu().numpy())
