#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import glob
import json
import random
import shutil
from tqdm import tqdm

import torch
import torchaudio

from soundspaces_nvas3d.utils.aihabitat_utils import load_room_grid
from soundspaces_nvas3d.utils.audio_utils import wiener_deconv_list
from nvas3d.utils.audio_utils import clip_two
from nvas3d.utils.utils import normalize, parse_librispeech_metadata, MP3D_SCENE_SPLITS
from nvas3d.utils.generate_dataset_utils import sample_speech, sample_nonspeech, sample_acoustic_guitar, sample_all, clip_source, load_ir_source_receiver, save_audio_list, compute_reverb

SOURCE1_DATA = 'all'  # speech, nonspeech, guitar
SOURCE2_DATA = 'all'

random.seed(42)

DATASET_NAME = f'nvas3d_square_{SOURCE1_DATA}_{SOURCE2_DATA}'
os.makedirs(f'data/{DATASET_NAME}', exist_ok=True)

grid_distance = 1.0
grid_distance_str = str(grid_distance).replace(".", "_")
target_shape_t = 256
ir_length = 72000
ir_clip_idx = ir_length - 1
hop_length = 480
len_clip = hop_length * (target_shape_t - 1) + ir_length - 1
sample_rate = 48000
snr = 100
audio_format = 'flac'

for split in ['train', 'val', 'test', 'demo']:
   # LibriSpeech
    if split == 'train':
        librispeech_dir = f'data/MIDI/clip/Speech/LibriSpeech48k/train'
    elif split == 'val':
        librispeech_dir = f'data/MIDI/clip/Speech/LibriSpeech48k/validation'
    elif split == 'test':
        librispeech_dir = f'data/MIDI/clip/Speech/LibriSpeech48k/test'
    else:
        librispeech_dir = f'data/MIDI/clip/Speech/LibriSpeech48k/validation'
    files_librispeech = glob.glob(librispeech_dir + '/**/*.flac', recursive=True)
    librispeech_metadata = parse_librispeech_metadata(f'data/MIDI/clip/Speech/LibriSpeech48k/SPEAKERS.TXT')

    # MIDI
    if split == 'train':
        all_instruments_dir = [path for path in glob.glob(os.path.join('data/MIDI/clip', '*/*', 'train')) if os.path.isdir(path)]
    elif split == 'val':
        all_instruments_dir = [path for path in glob.glob(os.path.join('data/MIDI/clip', '*/*', 'validation')) if os.path.isdir(path)]
    elif split == 'test':
        all_instruments_dir = [path for path in glob.glob(os.path.join('data/MIDI/clip', '*/*', 'test')) if os.path.isdir(path)]
    else:
        all_instruments_dir = [path for path in glob.glob(os.path.join('data/MIDI/clip', '*/*', 'validation')) if os.path.isdir(path)]

    # RIR
    ir_dir = f'data/nvas3d_square/ir/{split}/grid_{grid_distance_str}'

    # Image
    dirname_sourceimage = f'data/nvas3d_square/image/{split}/grid_{grid_distance_str}'

    # Iterate over rooms
    for i_room, room in enumerate(tqdm(MP3D_SCENE_SPLITS[split])):
        grid_points = load_room_grid(room, grid_distance)['grid_points']
        total_pairs = []
        filename = f'data/nvas3d_square/metadata/grid_{grid_distance_str}/{room}_square.json'
        with open(filename, 'r') as file:
            square_data = json.load(file)
        pairs = square_data['selected_pairs']
        # Add each pair with room id to the total list
        for pair in pairs:
            total_pairs.append((room,) + tuple(pair))

        random.shuffle(total_pairs)

        for i_pair, pair in enumerate(tqdm(total_pairs)):
            dirname = f'data/{DATASET_NAME}/{split}/{room}/{i_pair}'
            # os.makedirs(dirname, exist_ok=True)

            room, source_idx_list, receiver_idx_list, novel_receiver_idx = pair

            # Compute source
            if SOURCE1_DATA == 'speech':
                source1_audio, source1_class = sample_speech(files_librispeech, librispeech_metadata)
            elif SOURCE1_DATA == 'nonspeech':
                source1_audio, source1_class = sample_nonspeech(all_instruments_dir)
            elif SOURCE1_DATA == 'guitar':
                source1_audio, source1_class = sample_acoustic_guitar(all_instruments_dir)
            else:
                source1_audio, source1_class = sample_all(all_instruments_dir, librispeech_metadata)

            if SOURCE2_DATA == 'speech':
                source2_audio, source2_class = sample_speech(files_librispeech, librispeech_metadata)
            elif SOURCE2_DATA == 'nonspeech':
                source2_audio, source2_class = sample_nonspeech(all_instruments_dir)
            elif SOURCE2_DATA == 'guitar':
                source2_audio, source2_class = sample_acoustic_guitar(all_instruments_dir)
            else:
                source2_audio, source2_class = sample_all(all_instruments_dir, librispeech_metadata)

            source1_audio, source2_audio = clip_two(source1_audio, source2_audio)
            if not split == 'test':
                source1_audio, source2_audio = clip_source(source1_audio, source2_audio, len_clip)

            source1_audio = normalize(source1_audio)
            source2_audio = normalize(source2_audio)

            if torch.isnan(source1_audio).any() or torch.isnan(source2_audio).any():
                continue

            if split == 'test':
                if source1_audio.shape[0] < sample_rate * 10:  # skip for short (<10s)
                    continue

            os.makedirs(f'{dirname}/source', exist_ok=True)
            torchaudio.save(f'{dirname}/source/source1.{audio_format}', source1_audio[ir_clip_idx:].unsqueeze(0), sample_rate)
            torchaudio.save(f'{dirname}/source/source2.{audio_format}', source2_audio[ir_clip_idx:].unsqueeze(0), sample_rate)

            # Save IR
            os.makedirs(f'{dirname}/ir_receiver', exist_ok=True)
            ir1_list = load_ir_source_receiver(ir_dir, room, source_idx_list[0], receiver_idx_list, ir_length)
            ir2_list = load_ir_source_receiver(ir_dir, room, source_idx_list[1], receiver_idx_list, ir_length)
            ir3_list = load_ir_source_receiver(ir_dir, room, source_idx_list[2], receiver_idx_list, ir_length)
            save_audio_list(f'{dirname}/ir_receiver/ir1', ir1_list, sample_rate, audio_format)
            save_audio_list(f'{dirname}/ir_receiver/ir2', ir2_list, sample_rate, audio_format)
            save_audio_list(f'{dirname}/ir_receiver/ir3', ir3_list, sample_rate, audio_format)

            os.makedirs(f'{dirname}/ir_novel', exist_ok=True)
            ir1_novel = load_ir_source_receiver(ir_dir, room, source_idx_list[0], [novel_receiver_idx], ir_length)[0]
            ir2_novel = load_ir_source_receiver(ir_dir, room, source_idx_list[1], [novel_receiver_idx], ir_length)[0]
            torchaudio.save(f'{dirname}/ir_novel/ir1_novel.{audio_format}', ir1_novel.unsqueeze(0), sample_rate)
            torchaudio.save(f'{dirname}/ir_novel/ir2_novel.{audio_format}', ir2_novel.unsqueeze(0), sample_rate)

            # Save reverb
            os.makedirs(f'{dirname}/reverb', exist_ok=True)
            reverb1_list = compute_reverb(source1_audio, ir1_list)
            reverb2_list = compute_reverb(source2_audio, ir2_list)
            save_audio_list(f'{dirname}/reverb/reverb1', reverb1_list, sample_rate, audio_format)
            save_audio_list(f'{dirname}/reverb/reverb2', reverb2_list, sample_rate, audio_format)

            # Compute receiver
            os.makedirs(f'{dirname}/receiver', exist_ok=True)
            receiver_list = [reverb1 + reverb2 for reverb1, reverb2 in zip(reverb1_list, reverb2_list)]
            save_audio_list(f'{dirname}/receiver/receiver', receiver_list, sample_rate, audio_format)

            # Save Weiner
            os.makedirs(f'{dirname}/wiener', exist_ok=True)
            wiener1_list = wiener_deconv_list(receiver_list, ir1_list, snr)
            wiener2_list = wiener_deconv_list(receiver_list, ir2_list, snr)
            wiener3_list = wiener_deconv_list(receiver_list, ir3_list, snr)
            save_audio_list(f'{dirname}/wiener/wiener1', wiener1_list, sample_rate, audio_format)
            save_audio_list(f'{dirname}/wiener/wiener2', wiener2_list, sample_rate, audio_format)
            save_audio_list(f'{dirname}/wiener/wiener3', wiener3_list, sample_rate, audio_format)

            # Copy image
            os.makedirs(f'{dirname}/image', exist_ok=True)

            if source1_class.lower() == 'male' or source1_class.lower() == 'female':
                source1_class_render = source1_class.lower()
            else:
                source1_class_render = 'guitar'

            if source2_class.lower() == 'male' or source2_class.lower() == 'female':
                source2_class_render = source2_class.lower()
            else:
                source2_class_render = 'guitar'
            rgb1 = f'{dirname_sourceimage}/{room}/rgb_{room}_{source1_class_render}_{source_idx_list[0]}.npy'
            rgb2 = f'{dirname_sourceimage}/{room}/rgb_{room}_{source2_class_render}_{source_idx_list[1]}.npy'
            depth1 = f'{dirname_sourceimage}/{room}/depth_{room}_{source1_class_render}_{source_idx_list[0]}.npy'
            depth2 = f'{dirname_sourceimage}/{room}/depth_{room}_{source2_class_render}_{source_idx_list[1]}.npy'

            rgb1_bg = f'{dirname_sourceimage}/{room}/rgb_{room}_{source_idx_list[0]}.npy'
            rgb2_bg = f'{dirname_sourceimage}/{room}/rgb_{room}_{source_idx_list[1]}.npy'
            rgb3_bg = f'{dirname_sourceimage}/{room}/rgb_{room}_{source_idx_list[2]}.npy'
            depth1_bg = f'{dirname_sourceimage}/{room}/depth_{room}_{source_idx_list[0]}.npy'
            depth2_bg = f'{dirname_sourceimage}/{room}/depth_{room}_{source_idx_list[1]}.npy'
            depth3_bg = f'{dirname_sourceimage}/{room}/depth_{room}_{source_idx_list[2]}.npy'

            shutil.copy(rgb1, f'{dirname}/image/rgb1.npy')
            shutil.copy(rgb2, f'{dirname}/image/rgb2.npy')
            shutil.copy(depth1, f'{dirname}/image/depth1.npy')
            shutil.copy(depth2, f'{dirname}/image/depth2.npy')

            shutil.copy(rgb1_bg, f'{dirname}/image/rgb1_bg.npy')
            shutil.copy(rgb2_bg, f'{dirname}/image/rgb2_bg.npy')
            shutil.copy(rgb3_bg, f'{dirname}/image/rgb3_bg.npy')
            shutil.copy(depth1_bg, f'{dirname}/image/depth1_bg.npy')
            shutil.copy(depth2_bg, f'{dirname}/image/depth2_bg.npy')
            shutil.copy(depth3_bg, f'{dirname}/image/depth3_bg.npy')

            # # png (optional)
            # rgb1 = f'{dirname_sourceimage}/{room}/rgb_{room}_{source1_class_render}_{source_idx_list[0]}.png'
            # rgb2 = f'{dirname_sourceimage}/{room}/rgb_{room}_{source2_class_render}_{source_idx_list[1]}.png'
            # depth1 = f'{dirname_sourceimage}/{room}/depth_{room}_{source1_class_render}_{source_idx_list[0]}.png'
            # depth2 = f'{dirname_sourceimage}/{room}/depth_{room}_{source2_class_render}_{source_idx_list[1]}.png'

            # rgb1_bg = f'{dirname_sourceimage}/{room}/rgb_{room}_{source_idx_list[0]}.png'
            # rgb2_bg = f'{dirname_sourceimage}/{room}/rgb_{room}_{source_idx_list[1]}.png'
            # rgb3_bg = f'{dirname_sourceimage}/{room}/rgb_{room}_{source_idx_list[2]}.png'
            # depth1_bg = f'{dirname_sourceimage}/{room}/depth_{room}_{source_idx_list[0]}.png'
            # depth2_bg = f'{dirname_sourceimage}/{room}/depth_{room}_{source_idx_list[1]}.png'
            # depth3_bg = f'{dirname_sourceimage}/{room}/depth_{room}_{source_idx_list[2]}.png'

            # shutil.copy(rgb1, f'{dirname}/image/rgb1.png')
            # shutil.copy(rgb2, f'{dirname}/image/rgb2.png')
            # shutil.copy(depth1, f'{dirname}/image/depth1.png')
            # shutil.copy(depth2, f'{dirname}/image/depth2.png')

            # shutil.copy(rgb1_bg, f'{dirname}/image/rgb1_bg.png')
            # shutil.copy(rgb2_bg, f'{dirname}/image/rgb2_bg.png')
            # shutil.copy(rgb3_bg, f'{dirname}/image/rgb3_bg.png')
            # shutil.copy(depth1_bg, f'{dirname}/image/depth1_bg.png')
            # shutil.copy(depth2_bg, f'{dirname}/image/depth2_bg.png')
            # shutil.copy(depth3_bg, f'{dirname}/image/depth3_bg.png')

            # Save metadata
            metadata = {
                'source1_idx': source_idx_list[0],
                'source2_idx': source_idx_list[1],
                'source3_idx': source_idx_list[2],
                'receiver_idx_list': receiver_idx_list,
                'novel_receiver_idx': novel_receiver_idx,
                'source1_class': source1_class,
                'source2_class': source2_class,
                'grid_points': grid_points,
                'room': room,
            }
            torch.save(metadata, f'{dirname}/metadata.pt')

            pass
