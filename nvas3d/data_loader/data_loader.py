#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import random
import numpy as np

import torch
import torchvision
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from nvas3d.utils.utils import source_class_map

C = 343  # speed of sound


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SSAVDataset(Dataset):  # For training
    def __init__(self, split, use_visual, use_deconv, num_receivers, hop_length, win_length, n_fft, nvas3d_dataset, audio_format):
        random.seed(42)

        self.split = split
        self.use_visual = use_visual
        self.use_deconv = use_deconv
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.num_receivers = num_receivers
        self.nvas3d_dataset = nvas3d_dataset
        self.audio_format = audio_format

        # dataset_dir
        if split == 'train':
            self.dataset_dir = f'data/{self.nvas3d_dataset}/train'
        elif split == 'val':
            self.dataset_dir = f'data/{self.nvas3d_dataset}/val'
        else:
            self.dataset_dir = f'data/{self.nvas3d_dataset}/test'

        # Make room-idx list
        self.room_id_pairs = []
        for room in os.listdir(self.dataset_dir):
            room_path = os.path.join(self.dataset_dir, room)
            # For each idx in room
            for id in os.listdir(room_path):
                self.room_id_pairs.append((room, id))

    def __len__(self):
        return len(self.room_id_pairs)

    def __getitem__(self, idx):
        sample = {}
        room, id = self.room_id_pairs[idx]
        data_dir = f'{self.dataset_dir}/{room}/{id}'

        # Set query: positive or negative
        if int(id) % 2 == 0:  # positive for even id
            sample['is_source'] = torch.ones(1)
            query_id = 1
        else:
            sample['is_source'] = torch.zeros(1)
            query_id = 3

        # Load source audios
        source1_audio, _ = torchaudio.load(f'{data_dir}/source/source1.{self.audio_format}')
        source2_audio, _ = torchaudio.load(f'{data_dir}/source/source2.{self.audio_format}')
        source1_audio = source1_audio[0]
        source2_audio = source2_audio[0]

        # Load receiver audios
        receiver_audio = self.load_audio_list(f'{data_dir}/receiver/receiver')

        # Load input audios (receiver or wiener)
        if self.use_deconv:
            input_audio = self.load_audio_list(f'{data_dir}/wiener/wiener{query_id}')
        else:
            input_audio = receiver_audio

        # Normalize (maximum of audios becomes 1)
        max_value = max(abs(input_audio).max(), abs(source1_audio).max())
        input_audio /= max_value
        source1_audio /= max_value
        source2_audio /= max_value
        sample['input_audio'] = input_audio
        sample['source1_audio'] = source1_audio
        sample['source2_audio'] = source2_audio

        # STFT
        sample['source1_stft'] = torch.stft(source1_audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                                            window=torch.hamming_window(self.win_length), pad_mode='constant', return_complex=False).permute(2, 0, 1)  # [2, F, N]
        sample['input_stft'] = self.stft_audio_list(input_audio)  # [n_receivers * 2, n_freq, n_frame]

        # Load metadata
        metadata = torch.load(f'{data_dir}/metadata.pt')
        sample['source1_class'] = torch.tensor(source_class_map[metadata['source1_class']])
        sample['source2_class'] = torch.tensor(source_class_map[metadata['source2_class']])

        # Images
        if self.use_visual:
            if query_id == 1:  # positive query
                rgb = np.load(f'{data_dir}/image/rgb{query_id}.npy')
            elif query_id == 3:  # negative query
                rgb = np.load(f'{data_dir}/image/rgb{query_id}_bg.npy')
            depth = np.load(f'{data_dir}/image/depth{query_id}_bg.npy')  # TODO: no instruments

        visual_sensors = []
        if self.use_visual:
            sample['rgb'] = torch.from_numpy(rgb[:, :, :3] / 255.0).permute(2, 0, 1)
            visual_sensors.append('rgb')
            sample['depth'] = torch.from_numpy((depth - depth.min()) / (depth.max() - depth.min())).unsqueeze(0)
            visual_sensors.append('depth')
            for visual_sensor in visual_sensors:
                sample[visual_sensor] = torchvision.transforms.Resize((192, 576))(sample[visual_sensor])

        # For evaluation
        reverb1_audio, _ = torchaudio.load(f'{data_dir}/reverb/reverb1_1.{self.audio_format}')
        reverb2_audio, _ = torchaudio.load(f'{data_dir}/reverb/reverb2_1.{self.audio_format}')
        reverb1_audio /= max_value
        reverb2_audio /= max_value
        receiver_audio /= max_value
        sample['reverb1_audio'] = reverb1_audio
        sample['reverb2_audio'] = reverb2_audio
        sample['receiver_audio'] = receiver_audio

        ir1 = self.load_audio_list(f'{data_dir}/ir_receiver/ir1')
        sample['ir1'] = ir1

        return sample, room, id

    def load_audio_list(self, audio_list_name):
        """
        Helper function to load list of audios
        """

        audio_list = []
        for m in range(self.num_receivers):
            audio, _ = torchaudio.load(f'{audio_list_name}_{m+1}.{self.audio_format}')
            audio_list.append(audio)
        return torch.stack(audio_list, dim=0).reshape(self.num_receivers, -1)

    def stft_audio_list(self, audio_list):
        """
        Helper function to apply STFT to list of audios 
        """

        stft_list = []
        for m in range(self.num_receivers):
            audio_stft = torch.stft(audio_list[m], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                                      window=torch.hamming_window(self.win_length), pad_mode='constant', return_complex=False).permute(2, 0, 1)
            stft_list.append(audio_stft)
        _, n_freq, n_frame = audio_stft.shape
        return torch.stack(stft_list, dim=0).reshape(self.num_receivers * 2, n_freq, n_frame)


class SSAVDataLoader:
    def __init__(self, use_visual, use_deconv, is_ddp, batch_size, num_workers, num_receivers, hop_length, win_length, n_fft, nvas3d_dataset='nvas3d_square_all_all', audio_format='flac', test_mode=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        if not test_mode:
            self.train_dataset = SSAVDataset('train', use_visual, use_deconv, num_receivers, hop_length, win_length, n_fft, nvas3d_dataset, audio_format)
        self.val_dataset = SSAVDataset('val', use_visual, use_deconv, num_receivers, hop_length, win_length, n_fft, nvas3d_dataset, audio_format)
        self.test_dataset = SSAVDataset('test', use_visual, use_deconv, num_receivers, hop_length, win_length, n_fft, nvas3d_dataset, audio_format)

        self.is_ddp = is_ddp
        if self.is_ddp:
            if not test_mode:
                self.train_sampler = DistributedSampler(self.train_dataset)
            self.val_sampler = DistributedSampler(self.val_dataset)
            self.test_sampler = DistributedSampler(self.test_dataset)

    def get_train_data(self):
        if self.is_ddp:
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                sampler=self.train_sampler,
                pin_memory=True,
                num_workers=self.num_workers,
                worker_init_fn=seed_worker
            )
        else:
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.num_workers,
                worker_init_fn=seed_worker
            )

    def get_val_data(self):
        if self.is_ddp:
            return DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size,
                sampler=self.val_sampler,
                pin_memory=True,
                num_workers=self.num_workers,
                worker_init_fn=seed_worker
            )
        else:
            return DataLoader(
                dataset=self.val_dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
                worker_init_fn=seed_worker
            )

    def get_test_data(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker
        )


class SSAVDatasetQueryAll(Dataset):  # For demo
    def __init__(self, split, use_visual, use_deconv, num_receivers, hop_length, win_length, n_fft, nvas3d_dataset, audio_format, grid_distance=1.0):
        random.seed(42)

        self.split = split
        self.use_visual = use_visual
        self.use_deconv = use_deconv
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.num_receivers = num_receivers
        self.nvas3d_dataset = nvas3d_dataset
        self.audio_format = audio_format
        self.grid_distance = grid_distance

        # dataset_dir
        self.dataset_dir = f'data/{self.nvas3d_dataset}/{split}'

        # Make room-idx list
        self.room_id_pairs = []
        for room in os.listdir(self.dataset_dir):
            room_path = os.path.join(self.dataset_dir, room)
            # For each idx in room
            for id in os.listdir(room_path):
                if not id == 'image' and not id[-3:] == 'png':
                    self.room_id_pairs.append((room, id))

    def __len__(self):
        return len(self.room_id_pairs)

    def __getitem__(self, idx):
        sample = {}
        room, id = self.room_id_pairs[idx]
        data_dir = f'{self.dataset_dir}/{room}/{id}'

        # Load metadata
        metadata = torch.load(f'{data_dir}/metadata.pt')
        source1_idx = metadata['source1_idx']
        source2_idx = metadata['source2_idx']
        source_idx_list = [source1_idx, source2_idx]
        receiver_idx_list = metadata['receiver_idx_list']
        grid_points = metadata['grid_points']
        source1_class = metadata['source1_class']
        source2_class = metadata['source1_class']

        # Load source audios
        source1_audio, _ = torchaudio.load(f'{data_dir}/source/source1.{self.audio_format}')
        source2_audio, _ = torchaudio.load(f'{data_dir}/source/source2.{self.audio_format}')
        source1_audio = source1_audio[0]
        source2_audio = source2_audio[0]
        sample['source1_audio'] = source1_audio
        sample['source2_audio'] = source2_audio

        # Load receiver audios
        receiver_audio = self.load_audio_list(f'{data_dir}/receiver/receiver')
        sample['receiver_audio'] = receiver_audio
        receiver_stft = self.stft_audio_list(receiver_audio)  # [n_receivers * 2, n_freq, n_frame]

        # Load queries
        num_points = grid_points.shape[0]
        query_stft = []
        query_rgb = []
        query_depth = []
        query_idx_list = []
        query_positive = []
        query_audio = []

        for query_idx in range(num_points):
            if (query_idx in receiver_idx_list):
                continue

            query_idx_list.append(query_idx)

            # load Weiner
            if self.use_deconv:
                wiener_audio = self.load_audio_list(f'{data_dir}/wiener/wiener{query_idx}')
                query_audio.append(wiener_audio)
                wiener_stft = self.stft_audio_list(wiener_audio)  # [n_receivers * 2, n_freq, n_frame]
                query_stft.append(wiener_stft)
            else:
                query_stft.append(receiver_stft)

            # load image
            query_positive.append(query_idx in source_idx_list)
            image_dir = f'{self.dataset_dir}/{room}/image'
            if self.use_visual:
                if source1_class.lower() == 'male' or source1_class.lower() == 'female':
                    source1_class_render = source1_class.lower()
                else:
                    source1_class_render = 'guitar'
                if source2_class.lower() == 'male' or source2_class.lower() == 'female':
                    source2_class_render = source2_class.lower()
                else:
                    source2_class_render = 'guitar'

                if query_idx == source1_idx:
                    rgb = np.load(f'{image_dir}/rgb_{room}_{source1_class_render}_{query_idx}.npy')
                    depth = np.load(f'{image_dir}/depth_{room}_{query_idx}.npy')  # TODO: instruments mesh
                elif query_idx == source2_idx:
                    rgb = np.load(f'{image_dir}/rgb_{room}_{source2_class_render}_{query_idx}.npy')
                    depth = np.load(f'{image_dir}/depth_{room}_{query_idx}.npy')
                else:
                    rgb = np.load(f'{image_dir}/rgb_{room}_{query_idx}.npy')
                    depth = np.load(f'{image_dir}/depth_{room}_{query_idx}.npy')

            if self.use_visual:
                rgb = torch.from_numpy(rgb[:, :, :3] / 255.0).permute(2, 0, 1)
                rgb = torchvision.transforms.Resize((192, 576))(rgb)
                query_rgb.append(rgb)

                depth = torch.from_numpy((depth - depth.min()) / (depth.max() - depth.min())).unsqueeze(0)
                depth = torchvision.transforms.Resize((192, 576))(depth)
                query_depth.append(depth)
            else:
                query_rgb.append(0)  # placeholder
                query_depth.append(0)

        # For reverberant ss evaluation
        ir1 = self.load_audio_list(f'{data_dir}/ir_receiver/ir1')
        sample['ir1'] = ir1
        ir2 = self.load_audio_list(f'{data_dir}/ir_receiver/ir2')
        sample['ir2'] = ir2
        reverb1_audio, _ = torchaudio.load(f'{data_dir}/reverb/reverb1_1.{self.audio_format}')
        reverb2_audio, _ = torchaudio.load(f'{data_dir}/reverb/reverb2_1.{self.audio_format}')
        sample['reverb1_audio'] = reverb1_audio
        sample['reverb2_audio'] = reverb2_audio

        return sample, metadata, room, id, query_stft, query_rgb, query_depth, query_idx_list, query_positive, query_audio

    def load_audio_list(self, audio_list_name):
        """
        Helper function to load list of audios
        """

        audio_list = []
        for m in range(self.num_receivers):
            audio, _ = torchaudio.load(f'{audio_list_name}_{m+1}.{self.audio_format}')
            audio_list.append(audio)
        return torch.stack(audio_list, dim=0).reshape(self.num_receivers, -1)

    def stft_audio_list(self, audio_list):
        """
        Helper function to apply STFT to list of audios 
        """

        stft_list = []
        for m in range(self.num_receivers):
            audio_stft = torch.stft(audio_list[m], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                                      window=torch.hamming_window(self.win_length), pad_mode='constant', return_complex=False).permute(2, 0, 1)
            stft_list.append(audio_stft)
        _, n_freq, n_frame = audio_stft.shape
        return torch.stack(stft_list, dim=0).reshape(self.num_receivers * 2, n_freq, n_frame)
