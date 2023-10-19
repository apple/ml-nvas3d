#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import glob
import json
import random
import subprocess
import concurrent.futures
from itertools import product

from tqdm import tqdm

import torch
import torchaudio

from soundspaces_nvas3d.utils.aihabitat_utils import load_room_grid
from soundspaces_nvas3d.utils.audio_utils import wiener_deconv_list
from nvas3d.utils.audio_utils import clip_two
from nvas3d.utils.utils import normalize, parse_librispeech_metadata, MP3D_SCENE_SPLITS
from nvas3d.utils.generate_dataset_utils import sample_speech, sample_nonspeech, sample_acoustic_guitar, sample_all, sample_instrument, clip_source, load_ir_source_receiver, save_audio_list, compute_reverb

os.makedirs('data/temp', exist_ok=True)


SOURCE1_DATA = 'Guitar'
SOURCE2_DATA = 'Guitar'

num_id_per_room = 1

random.seed(42)

DATASET_NAME = f'nvas3d_square_{SOURCE1_DATA}_{SOURCE2_DATA}_queryall_{num_id_per_room}_v3'
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

for split in ['val']:
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
    if split == 'val_trainscene':
        split_scene = 'train'
    else:
        split_scene = split
    ir_dir = f'data/nvas3d_square/ir/{split_scene}/grid_{grid_distance_str}'

    # Image
    dirname_sourceimage = f'data/nvas3d_square/image/{split_scene}/grid_{grid_distance_str}'

    # Iterate over rooms
    for i_room, room in enumerate(tqdm(MP3D_SCENE_SPLITS[split_scene])):
        grid_points = load_room_grid(room, grid_distance)['grid_points']
        num_points = grid_points.shape[0]
        total_pairs = []
        filename = f'data/nvas3d_square/metadata/grid_{grid_distance_str}/{room}_square.json'  # from generate_metadata_square.json
        with open(filename, 'r') as file:
            square_data = json.load(file)
        pairs_all = square_data['selected_pairs']
        # Add each pair with room id to the total list

        random.shuffle(pairs_all)

        # pairs = pairs[:num_id_per_room]

        pairs = []
        for pair in pairs_all:
            source_idx_list, receiver_idx_list, novel_receiver_idx = pair
            if (novel_receiver_idx not in source_idx_list) and (novel_receiver_idx not in receiver_idx_list):
                pairs.append(pair)
            # else:
            #     print(f'invalid idx: {source_idx_list}, {receiver_idx_list}, {novel_receiver_idx}')
            if len(pairs) >= num_id_per_room:
                break

        # All IRs
        # Initialize a list to store all combinations
        all_combinations = []

        # Iterate over selected pairs
        for pair in pairs:
            # Unpack the pair
            _, receiver_idxs, _ = pair
            # Get all combinations of source and receiver indices
            comb = product(list(range(num_points)), receiver_idxs)
            # Add these combinations to the list
            all_combinations.extend(comb)
        all_combinations = list(set(all_combinations))  # remove redundancy

        # download wav files # Replace to render IR
        # temp_list = set()
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     for source_idx in executor.map(download_wav, all_combinations):
        #         temp_list.add(source_idx)
        # temp_list = list(temp_list)

        # Render image
        dirname_target_image = f'data/{DATASET_NAME}/{split}/{room}/image'
        os.makedirs(dirname_target_image, exist_ok=True)
        query_idx_list = list(range(num_points))
        subprocess.run(['python', 'soundspaces_nvas3d/image_rendering/generate_target_image.py', '--room', room, '--dirname', dirname_target_image, '--source_idx_list', ' '.join(map(str, query_idx_list))])

        # For each pair, make data
        for i_pair, pair in enumerate(tqdm(pairs)):
            dirname = f'data/{DATASET_NAME}/{split}/{room}/{i_pair}'
            source_idx_list, receiver_idx_list, novel_receiver_idx = pair

            os.makedirs(dirname, exist_ok=True)

            # Compute source
            os.makedirs(f'{dirname}/source', exist_ok=True)
            if SOURCE1_DATA == 'speech':
                source1_audio, source1_class = sample_speech(files_librispeech, librispeech_metadata)
            elif SOURCE1_DATA == 'nonspeech':
                source1_audio, source1_class = sample_nonspeech(all_instruments_dir)
            elif SOURCE1_DATA == 'guitar':
                source1_audio, source1_class = sample_acoustic_guitar(all_instruments_dir)
            elif SOURCE1_DATA == 'all':
                source1_audio, source1_class = sample_all(all_instruments_dir, librispeech_metadata)
            else:
                source1_audio, source1_class = sample_instrument(all_instruments_dir, librispeech_metadata, SOURCE1_DATA)

            if SOURCE2_DATA == 'speech':
                source2_audio, source2_class = sample_speech(files_librispeech, librispeech_metadata)
            elif SOURCE2_DATA == 'nonspeech':
                source2_audio, source2_class = sample_nonspeech(all_instruments_dir)
            elif SOURCE2_DATA == 'guitar':
                source2_audio, source2_class = sample_acoustic_guitar(all_instruments_dir)
            elif SOURCE2_DATA == 'all':
                source2_audio, source2_audio = sample_all(all_instruments_dir, librispeech_metadata)
            else:
                source2_audio, source2_class = sample_instrument(all_instruments_dir, librispeech_metadata, SOURCE2_DATA)

            source1_audio, source2_audio = clip_two(source1_audio, source2_audio)
            if not (split == 'test' or split == 'val'):  # check
                source1_audio, source2_audio = clip_source(source1_audio, source2_audio, len_clip)
            source1_audio = normalize(source1_audio)
            source2_audio = normalize(source2_audio)

            if torch.isnan(source1_audio).any() or torch.isnan(source2_audio).any():
                continue

            if split == 'test' or split == 'val':
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

            # Compute novel
            os.makedirs(f'{dirname}/receiver_novel', exist_ok=True)
            reverb1_novel = compute_reverb(source1_audio, [ir1_novel])[0]
            reverb2_novel = compute_reverb(source2_audio, [ir2_novel])[0]
            receiver_novel = reverb1_novel + reverb2_novel
            torchaudio.save(f'{dirname}/receiver_novel/receiver_novel.{audio_format}', receiver_novel.unsqueeze(0), sample_rate)

            # # Render binaural IR and audio
            # if not os.path.exists(f'{dirname}/ir_novel_binaural'):
            #     os.makedirs(f'{dirname}/ir_novel_binaural', exist_ok=True)
            #     ir_novel_list, source_idx_pair_list, receiver_idx_pair_list = render_ir_parallel_room_idx(room, source_idx_list, [novel_receiver_idx], filename=None, use_default_material=False, channel_type='Binaural')
            #     for idx_novel, ir_novel in enumerate(ir_novel_list[:2]):  # first two sources
            #         if ir_novel[0].shape[0] > ir_length:
            #             ir_binaural = ir_novel[:][:ir_length]
            #         else:
            #             ir_binaural = F.pad(ir_novel[:], (0, ir_length - ir_novel.shape[1]))
            #         torchaudio.save(f'{dirname}/ir_novel_binaural/ir{idx_novel+1}_novel_binaural.{audio_format}', ir_binaural, sample_rate)

            # # set image rendering
            # all_receiver_idx_list = receiver_idx_list.copy()
            # all_receiver_idx_list.append(novel_receiver_idx)

            # def run_function(function_name, *args):  # use subprocess because of memory leak in soundspaces
            #     subprocess.run(["python", "script/render_scene_script.py", function_name, *map(str, args)])

            # # Render novel-view RGBD image
            # os.makedirs(f'{dirname}/image_receiver', exist_ok=True)
            # source_class_list = [source1_class, source2_class]
            # run_function("receiver", f'{dirname}/image_receiver', room, source_idx_list[:2], source_class_list, all_receiver_idx_list)

            # # Save topdown view
            # run_function("scene", f'{dirname}/config.png', room, source_idx_list[:2], all_receiver_idx_list)

            # Save metadata
            metadata = {
                'source1_idx': source_idx_list[0],
                'source2_idx': source_idx_list[1],
                'receiver_idx_list': receiver_idx_list,
                'novel_receiver_idx': novel_receiver_idx,
                'source1_class': source1_class,
                'source2_class': source2_class,
                'grid_points': grid_points,
                'room': room,
            }

            torch.save(metadata, f'{dirname}/metadata.pt')

            # Query
            for query_idx in range(num_points):
                if (query_idx in receiver_idx_list):
                    continue

                # Load IR
                ir_query_list = load_ir_source_receiver(f'data/temp/ir', room, query_idx, receiver_idx_list, ir_length)

                # Save Weiner
                os.makedirs(f'{dirname}/wiener', exist_ok=True)
                wiener_list = wiener_deconv_list(receiver_list, ir_query_list, snr)

                save_audio_list(f'{dirname}/wiener/wiener{query_idx}', wiener_list, sample_rate, audio_format)

                # # debug delay
                # delay_ir = torch.nonzero(ir_query_list[0], as_tuple=True)[0][0]
                # receiver_point = grid_points[receiver_idx_list[0]]
                # query_point = grid_points[query_idx]

                # c = 343
                # dist = (query_point - receiver_point).norm() - 0.1  # soundspaces offset
                # delay_dist = (int(dist / c * sample_rate))
                # print(delay_ir, delay_dist)
                # pass
