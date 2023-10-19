#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import json
import random
import argparse
import subprocess
import typing as T

import torch
import torchaudio

from soundspaces_nvas3d.utils.ss_utils import render_ir_parallel_room_idx, create_scene
from soundspaces_nvas3d.utils.aihabitat_utils import load_room_grid
from soundspaces_nvas3d.utils.audio_utils import wiener_deconv_list
from nvas3d.utils.audio_utils import clip_two
from nvas3d.utils.utils import normalize
from nvas3d.utils.generate_dataset_utils import load_ir_source_receiver, save_audio_list, compute_reverb


def generate_rir(
    args: argparse.Namespace,
    room: str,
    source_idx_list: T.List[int],
    receiver_idx_list: T.List[int]
):
    """
    Generates and saves Room Impulse Response (RIR) data for pairs of source_idx_list and receiver_idx_list.

    Args:
    - args:                 Parsed command line arguments for dirname and grid distance.
    - room:                 Name of the room.
    - source_idx_list:      List of source indices.
    - receiver_idx_list:    List of receiver indices.
    """

    ir_dir = f'data/{args.dataset_dir}/temp/ir/grid_{str(args.grid_distance).replace(".", "_")}'
    filename_ir = f'{ir_dir}/{room}/ir'
    os.makedirs(filename_ir, exist_ok=True)
    render_ir_parallel_room_idx(room, source_idx_list, receiver_idx_list, filename_ir, args.grid_distance)


def visualize_grid(
    args: argparse.Namespace,
    room: str,
    image_size: T.Tuple[int, int]
):
    """
    Visualizes grid points for a given room.

    Args:
    - args:         Parsed command line arguments for dirname and grid distance.
    - room:         Name of the room.
    - image_size:   Dimensions of the output image.
    """

    scene = create_scene(room, image_size=image_size)
    os.makedirs(f'data/{args.dataset_dir}/demo/{room}', exist_ok=True)
    scene.generate_xy_grid_points(args.grid_distance, filename_png=f'data/{args.dataset_dir}/demo/{room}/index_{str(args.grid_distance).replace(".", "_")}.png')
    scene.generate_xy_grid_points(0.5, filename_png=f'data/{args.dataset_dir}/demo/{room}/index_0_5.png')


def process_audio_sources(
    dirname: str,
    source1_path: str,
    source2_path: str,
    rir_clip_idx: int,
    sample_rate: int,
    audio_format: str
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess and saves audio for dry sound.

    Args:
    - dirname:          Directory name for saving the processed audio.
    - source1_path:     Path to the first audio source.
    - source2_path:     Path to the second audio source.
    - rir_clip_idx:     Index for clipping the RIR.
    - sample_rate:      Sampling rate of the audio.
    - audio_format:     Format to save the audio (e.g., 'flac', 'wav').

    Returns:
    - Processed audio tensors for source1 and source2.
    """

    source1_audio, _ = torchaudio.load(source1_path)
    source2_audio, _ = torchaudio.load(source2_path)
    source1_audio = source1_audio.reshape(-1)
    source2_audio = source2_audio.reshape(-1)
    source1_audio, source2_audio = clip_two(source1_audio, source2_audio)
    source1_audio = normalize(source1_audio)
    source2_audio = normalize(source2_audio)
    os.makedirs(f'{dirname}/source', exist_ok=True)
    torchaudio.save(f'{dirname}/source/source1.{audio_format}', source1_audio[rir_clip_idx:].unsqueeze(0), sample_rate)
    torchaudio.save(f'{dirname}/source/source2.{audio_format}', source2_audio[rir_clip_idx:].unsqueeze(0), sample_rate)

    return source1_audio, source2_audio


def save_rir_data(
    dirname: str,
    rir_dir: str,
    room: str,
    source_idx_list: T.List[int],
    receiver_idx_list: T.List[int],
    rir_length: int,
    sample_rate: int,
    audio_format: str
) -> T.Tuple[T.List[torch.Tensor], T.List[torch.Tensor]]:
    """
    Saves RIR data for the pair of sources and receivers.

    Args:
    - dirname:              Directory name for saving the RIR data.
    - rir_dir:              Directory where RIR data is located.
    - room:                 Name of the room.
    - source_idx_list:      List of source indices.
    - receiver_idx_list:    List of receiver indices.
    - rir_length:           Length of the RIR.
    - sample_rate:          Sampling rate of the audio.
    - audio_format:         Format to save the audio (e.g., 'flac', 'wav').

    Returns:
    - Lists of IRs from source1 and source2.
    """

    os.makedirs(f'{dirname}/ir_receiver', exist_ok=True)
    rir1_list = load_ir_source_receiver(rir_dir, room, source_idx_list[0], receiver_idx_list, rir_length)
    rir2_list = load_ir_source_receiver(rir_dir, room, source_idx_list[1], receiver_idx_list, rir_length)
    save_audio_list(f'{dirname}/ir_receiver/ir1', rir1_list, sample_rate, audio_format)
    save_audio_list(f'{dirname}/ir_receiver/ir2', rir2_list, sample_rate, audio_format)
    return rir1_list, rir2_list


def reverb_audio(
    filename: str,
    source_audio: torch.Tensor,
    rir_list: T.List[torch.Tensor],
    sample_rate: int,
    audio_format: str
) -> T.List[torch.Tensor]:
    """
    Applies reverberation to audio using provided IR.

    Args:
    - filename:          Directory name for saving the reverberant audio.
    - source_audio:     Source audio tensor.
    - rir_list:         List of RIR tensors.
    - sample_rate:      Sampling rate of the audio.
    - audio_format:     Format to save the audio (e.g., 'flac', 'wav').

    Returns:
    - List of reverberated audio.
    """

    reverb_list = compute_reverb(source_audio, rir_list)
    save_audio_list(filename, reverb_list, sample_rate, audio_format)
    return reverb_list


def mix_audio(
    dirname: str,
    reverb1_list: T.List[torch.Tensor],
    reverb2_list: T.List[torch.Tensor],
    sample_rate: int,
    audio_format: str
) -> T.List[torch.Tensor]:
    """
    Mixes two reverberant audio.

    Args:
    - dirname:          Directory name for saving the mixed audio.
    - reverb1_list:     List of the first reverberant audio.
    - reverb2_list:     List of the second reverberant audio.
    - sample_rate:      Sampling rate of the audio.
    - audio_format:     Format to save the audio (e.g., 'flac').

    Returns:
    - List containing mixed audio.
    """

    os.makedirs(f'{dirname}/receiver', exist_ok=True)
    receiver_list = [reverb1 + reverb2 for reverb1, reverb2 in zip(reverb1_list, reverb2_list)]
    save_audio_list(f'{dirname}/receiver/receiver', receiver_list, sample_rate, audio_format)
    return receiver_list


def save_topdown_view(
    function_name: str,
    args_list: T.List[T.Union[str, int, float]]
):
    """
    Saves a topdown view of a given scene.

    Args:
    - function_name:    Name of the function to be used for visualization.
    - args_list:        List of arguments for the visualization function.
                        (room, source_idx_list, receiver_idx_list, grid_distance)
    """

    subprocess.run(["python", "soundspaces_nvas3d/utils/render_scene_script.py", function_name, *map(str, args_list)])


def main(args):
    """
    Generate and save demo data for a specific room.

    Directory Structure and Contents:
    ├── {data_demo} = data/nvas3d_demo/demo/{room}/0
    │   ├── receiver/          : Receiver audio.
    │   ├── wiener/            : Deconvolved audio (auxiliary data to accelerate tests) (wiener_{query_idx}_{receiver_id}). 
    │   ├── source/            : Ground truth dry audio.
    │   ├── reverb1/           : Ground truth reverberant audio for source 1.    
    │   ├── reverb2/           : Ground truth reverberant audio for source 2.    
    │   ├── ir_receiver/       : Ground truth RIRs from source to receiver.    
    │   ├── config.png         : Visualization of room configuration.
    │   └── metadata.pt        : Metadata containing source indices, classes, grid points, and room information.

    Additional Visualizations:
    ├── data/nvas3d_demo/{room} : Room index visualizations.
    """

    # Seed for reproducibility
    random.seed(42)

    # Directory setup for data
    os.makedirs(f'data/{args.dataset_dir}', exist_ok=True)

    # Extract and load room and grid related data
    room = args.room
    grid_distance = args.grid_distance
    grid_points = load_room_grid(room, grid_distance)['grid_points']
    rir_length = args.rir_length
    sample_rate = args.sample_rate
    snr = args.snr
    audio_format = args.audio_format
    scene_config = args.scene_config

    source1_path = args.source1_path
    source2_path = args.source2_path

    # Set source and receiver locations
    with open(f'demo/config_demo/{scene_config}.json', 'r') as file:
        json_scene = json.load(file)
    source_idx_list = json_scene['source_idx_list']
    receiver_idx_list = json_scene['receiver_idx_list']

    # Generate and save RIR data
    generate_rir(args, room, list(range(len(grid_points))), receiver_idx_list)

    # Prepare directory for demo data
    dirname = f'data/{args.dataset_dir}/demo/{room}/0'
    os.makedirs(dirname, exist_ok=True)

    # Visualize the grid points within the room
    visualize_grid(args, room, (400, 300))

    # Preprocess audio
    rir_clip_idx = rir_length - 1
    source1_class, source2_class = 'female', 'drum'
    source1_audio, source2_audio = process_audio_sources(dirname, source1_path, source2_path, rir_clip_idx, sample_rate, audio_format)

    # Reverb audio for source 1 and source 2
    rir_dir = f'data/{args.dataset_dir}/temp/ir/grid_{str(grid_distance).replace(".", "_")}'
    rir1_list, rir2_list = save_rir_data(dirname, rir_dir, room, source_idx_list, receiver_idx_list, rir_length, sample_rate, audio_format)
    os.makedirs(f'{dirname}/reverb', exist_ok=True)
    reverb1_list = reverb_audio(f'{dirname}/reverb/reverb1', source1_audio, rir1_list, sample_rate, audio_format)
    reverb2_list = reverb_audio(f'{dirname}/reverb/reverb2', source2_audio, rir2_list, sample_rate, audio_format)

    # Mix both reverberant audios
    receiver_list = mix_audio(dirname, reverb1_list, reverb2_list, sample_rate, audio_format)

    # Visualize topdown view of the scene
    save_topdown_view("scene", [f'{dirname}/config.png', room, source_idx_list[:2], receiver_idx_list, args.grid_distance])

    # Save metadata
    metadata = {
        'source1_idx': source_idx_list[0],
        'source2_idx': source_idx_list[1],
        'receiver_idx_list': receiver_idx_list,
        'source1_class': source1_class,
        'source2_class': source2_class,
        'grid_points': grid_points,
        'grid_distance': grid_distance,
        'room': room,
    }
    torch.save(metadata, f'{dirname}/metadata.pt')

    # Save deconvolved audio using Wiener deconvolution
    for query_idx in range(len(grid_points)):
        if (query_idx in receiver_idx_list):
            continue

        # load RIR
        rir_query_list = load_ir_source_receiver(rir_dir, room, query_idx, receiver_idx_list, rir_length)

        # save Weiner
        os.makedirs(f'{dirname}/wiener', exist_ok=True)
        wiener_list = wiener_deconv_list(receiver_list, rir_query_list, snr)
        save_audio_list(f'{dirname}/wiener/wiener{query_idx}', wiener_list, sample_rate, audio_format)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--room', default='17DRP5sb8fy', type=str, help='mp3d room')
    parser.add_argument('--dataset_dir', default='nvas3d_demo', type=str, help='dirname')
    parser.add_argument('--grid_distance', default=1.0, type=float, help='Distance between grid points')
    parser.add_argument('--rir_length', default=72000, type=int, help='IR length')
    parser.add_argument('--sample_rate', default=48000, type=int, help='Sample rate')
    parser.add_argument('--snr', default=100, type=int, help='SNR for Wiener deconvolution')
    parser.add_argument('--audio_format', default='flac', type=str, help='Audio format to save')
    parser.add_argument('--scene_config', default='scene1_17DRP5sb8fy', type=str, help='Scene configuration json')
    parser.add_argument('--source1_path', default='data/source/female.flac', type=str, help='Filename for source 1')
    parser.add_argument('--source2_path', default='data/source/drum.flac', type=str, help='Filename for source 2')

    args = parser.parse_args()
    main(args)
