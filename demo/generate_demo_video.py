#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import json
import argparse
import itertools
import subprocess
import typing as T

import torch
import imageio
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import *

from nvas3d.utils.dynamic_utils import convolve_moving_receiver, setup_dynamic_interp
from nvas3d.utils.audio_utils import clip_two, clip_all
from soundspaces_nvas3d.utils.ss_utils import create_scene, render_rir_parallel
from soundspaces_nvas3d.utils.aihabitat_utils import load_room_grid
from soundspaces_nvas3d.soundspaces_nvas3d import Receiver, Source, Scene


def normalize(input: torch.Tensor) -> torch.Tensor:
    output = (input - input.min()) / (input.max() - input.min())
    output = 2 * output - 1

    return output


def configure_scene_from_metadata(
    metadata: T.Dict[str, T.Any],
    image_size: T.Tuple[int, int] = (1000, 1000),
    hfov: float = 90.0,
    use_placeholder_mesh: bool = False
) -> Scene:
    """
    Configures a scene using the provided metadata.

    Args:
    - metadata:                 Dictionary containing room and grid point information.
    - image_size:               The size of the rendered image.
    - hfov:                     Horizontal field of view.
    - use_placeholder_mesh:     Flag to determine if placeholder meshes should be used.

    Returns:
    - Configured scene object.
    """

    room = metadata['room'][0]
    grid_points_source = metadata['grid_points'][0]
    source_idx_list = [metadata['source1_idx'][0].item(), metadata['source2_idx'][0].item()]
    receiver_idx_list_original = torch.tensor(metadata['receiver_idx_list'])[:4]

    scene = create_scene(room, image_size=image_size, hfov=hfov)

    if use_placeholder_mesh:
        # Add placeholder mesh for sources and receivers to the scene
        # Download the following mesh objects and locate it under data/objects/{mesh_name}.glb:
        # - "Bluetooth Speaker" (https://skfb.ly/6VLyL) by Ramanan is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).
        # - “Classic Microphone” (https://skfb.ly/6Aryq) by urbanmasque is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/)
        # - "Standard Drum Set" (https://skfb.ly/owroB) by Heataker is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).
        # - "3D Posed People" (https://renderpeople.com/free-3d-people/) by Renderpeople: The licensing for our Renderpeople products includes that customers are allowed to use the data for rendering still images and animations for commercial or private purposes, such as video production, broadcasting, print, movies, advertising, illustrations and presentations (https://renderpeople.com/faq/)

        ss_source1 = Source(
            position=grid_points_source[source_idx_list[0]],
            rotation=0,
            dry_sound='',
            mesh='bluetooth_speaker',  # Need mesh object
            device=torch.device('cpu')
        )

        ss_source2 = Source(
            position=grid_points_source[source_idx_list[1]],
            rotation=-90,
            dry_sound='',
            mesh='bluetooth_speaker',  # Need mesh object
            device=torch.device('cpu')
        )

        ss_mic_list = [
            Source(
                position=grid_points_source[idx],
                rotation=180,
                dry_sound='',
                mesh='classic_microphone',  # Need mesh object
                device=torch.device('cpu')
            ) for idx in receiver_idx_list_original
        ]

        scene.add_source_mesh = True
        scene.source_list = [None] * (len(source_idx_list) + len(receiver_idx_list_original))
        scene.update_source(ss_source1, 0)
        scene.update_source(ss_source2, 1)

        for m, mic in enumerate(ss_mic_list):
            scene.update_source(mic, m + 2)

    return scene


def interpolate_moving_audio(
    source1_audio: torch.Tensor,
    source2_audio: torch.Tensor,
    ir1_list: T.List[torch.Tensor],
    ir2_list: T.List[torch.Tensor],
    receiver_position: torch.Tensor
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Interpolates audio for a moving receiver.

    Args:
    - source1_audio:        First source audio.
    - source2_audio:        Second source audio.
    - ir1_list:             List of impulse responses for source 1.
    - ir2_list:             List of impulse responses for source 2.
    - receiver_position:    Positions of the moving receiver.

    Returns:
    - Tuple containing combined audio, interpolated audio from source 1, and interpolated audio from source 2.
    """

    # Prepare for interpolation
    audio_len = source1_audio.shape[-1]
    interp_index, interp_weight = setup_dynamic_interp(receiver_position.numpy(), audio_len)

    # Generate audio for moving receiver
    receiver_audio_1 = convolve_moving_receiver(source1_audio.numpy()[0], ir1_list.numpy(), interp_index, interp_weight)
    receiver_audio_2 = convolve_moving_receiver(source2_audio.numpy()[0], ir2_list.numpy(), interp_index, interp_weight)
    receiver_audio_1 = receiver_audio_1[..., :source1_audio.shape[-1]]
    receiver_audio_2 = receiver_audio_2[..., :source1_audio.shape[-1]]

    # Mix and normalize audios
    receiver_audio = (receiver_audio_1 + receiver_audio_2)
    scale = np.max(abs(receiver_audio))
    receiver_audio /= scale
    receiver_audio_1 /= scale
    receiver_audio_2 /= scale

    return torch.from_numpy(receiver_audio), torch.from_numpy(receiver_audio_1), torch.from_numpy(receiver_audio_2)


def interpolate_rgb_images(
    scene: Scene,
    receiver_position: torch.Tensor,
    receiver_rotation_list: T.List[float],
    video_len: int
) -> T.List[np.ndarray]:
    """
    Interpolates RGB images based on receiver movement and rotation.

    Args:
    - scene:                  Scene object to render the images from.
    - receiver_position:      Positions of the receiver along the path.
    - receiver_rotation_list: List of rotations for the receiver.
    - video_len:              Number of frames in the video.

    Returns:
    - List of interpolated RGB images.
    """

    interp_index, interp_weight = setup_dynamic_interp(receiver_position.numpy(), video_len)

    interpolated_rgb_list = []

    for t in range(len(interp_index)):
        # Find the positions and rotations between which we're interpolating
        start_idx = interp_index[t]
        end_idx = start_idx + 1
        start_pos = receiver_position[start_idx]
        end_pos = receiver_position[end_idx]

        start_rot = receiver_rotation_list[start_idx]
        end_rot = receiver_rotation_list[end_idx]

        # Interpolate position and rotation
        receiver_position_interp = interpolate_values(start_pos, end_pos, interp_weight[t])
        receiver_rotation_interp = interpolate_values(start_rot, end_rot, interp_weight[t])

        receiver = Receiver(receiver_position_interp, receiver_rotation_interp)
        scene.update_receiver(receiver)

        rgb, _ = scene.render_image()
        interpolated_rgb_list.append(rgb[..., :3])

    return interpolated_rgb_list


def all_pairs(
    list1: T.List[T.Any],
    list2: T.List[T.Any]
) -> T.Tuple[T.List[T.Any], T.List[T.Any]]:
    """
    Computes all pairs of combinations between two lists.

    Args:
    - list1: First list.
    - list2: Second list.

    Returns:
    - Two lists containing paired elements from list1 and list2.
    """

    list_pair = list(itertools.product(list1, list2))

    list1_pair, list2_pair = zip(*list_pair)
    list1_pair = list(list1_pair)
    list2_pair = list(list2_pair)

    return list1_pair, list2_pair


def generate_rir_combination(
    room: str,
    source_idx_list: T.List[int],
    grid_points_source: torch.Tensor,
    receiver_idx_list: T.List[int],
    receiver_rotation_list: T.List[float],
    grid_points_receiver: torch.Tensor,
    channel_type: str = 'Binaural',
    channel_order: int = 0
) -> T.List[T.List[torch.Tensor]]:
    """
    Generates room impulse responses (RIR) for given source and receiver combinations.

    Args:
    - room:                     Room object for which RIRs need to be computed.
    - source_idx_list:          List of source indices.
    - grid_points_source:       Grid points for the source.
    - receiver_idx_list:        List of receiver indices.
    - receiver_rotation_list:   List of receiver rotations.
    - grid_points_receiver:     Grid points for the receiver.
    - channel_type:             Type of the channel. Defaults to 'Ambisonics'.
    - channel_order:            Order of the channel for Ambisonics. Defulats to 0, as video usually does not support HOA.

    Returns:
    - A 2D list containing RIRs for every source-receiver combination.
    """

    # Set source and receiver points
    source_point_list = grid_points_source[source_idx_list]
    receiver_point_list = grid_points_receiver[receiver_idx_list]

    source_points_pair, receiver_points_pair = all_pairs(source_point_list, receiver_point_list)
    _, receiver_rotation_pair = all_pairs(source_point_list, receiver_rotation_list)

    room_list = [room] * len(source_points_pair)
    filename_list = None

    # Render RIR for grid points
    ir_list = render_rir_parallel(room_list, source_points_pair, receiver_points_pair, receiver_rotation_list=receiver_rotation_pair, filename_list=filename_list, channel_type=channel_type, channel_order=channel_order)
    ir_list = clip_all(ir_list)  # make the length consistent
    num_channel = len(ir_list[0])

    # Reshape RIR
    num_sources = len(source_idx_list)
    num_receivers = len(receiver_idx_list)
    ir_output = torch.stack(ir_list).reshape(num_sources, num_receivers, num_channel, -1)  # '-1' will infer the remaining dimension based on the size of each tensor in ir_list
    ir_output /= ir_output.abs().max()

    return ir_output


def interpolate_values(
    start: float,
    end: float,
    interp_weight: float
) -> float:
    """
    Interpolate between two values based on the weight values.

    Args:
    - start:            Beginning value.
    - end:              Ending value.
    - interp_weight:    Weight for linear interpolation

    Returns:
    - Interpolated value.
    """

    return (1 - interp_weight) * start + interp_weight * end


def main(args):
    """
    Generate NVAS video from the estimated dry sound.

    Save:
    ├── {results_demo} = results/nvas3d_demo/default/demo/{room}/0
    │   ├── video/
    │   │   ├── moving_audio.wav    : Audio interpolated for the moving receiver.
    │   │   ├── moving_audio_1.wav  : Audio interpolated specifically for source 1.
    │   │   ├── moving_audio_2.wav  : Audio interpolated specifically for source 2.
    │   │   ├── moving_video.mp4    : Video visualization of movement (no audio).
    │   │   ├── nvas.mp4            : NVAS video results with combined audio.
    │   │   ├── nvas_source1.mp4    : NVAS video results for only source 1 audio.
    │   │   ├── nvas_source2.mp4    : NVAS video results for only source 2 audio.
    │   │   └── rgb_receiver.png    : A rendered view from the perspective of the receiver.
    """

    # Constants
    sample_rate = args.sample_rate
    sample_rate_video = args.sample_rate_video
    novel_path_config = args.novel_path_config
    use_gt_location = args.use_gt_location
    channel_type = args.channel_type
    use_placeholder_mesh = args.use_placeholder_mesh

    # Load data and metadata
    metadata = torch.load(f'{args.results_dir}/results_detection/metadata.pt')
    room = metadata['room'][0]
    grid_points_source = metadata['grid_points'][0]
    receiver_idx_list_original = torch.tensor(metadata['receiver_idx_list'])[:4]

    if use_gt_location:
        # Use estimated dry sound from GT source location
        source1_idx = metadata['source1_idx'][0].item()
        source2_idx = metadata['source2_idx'][0].item()
        source_idx_list = [source1_idx, source2_idx]
    else:
        # Use estimated dry sound from detected source location
        detected_source1_idx = metadata['detected_source_idx'][0]
        detected_source2_idx = metadata['detected_source_idx'][1]
        source_idx_list = [detected_source1_idx, detected_source2_idx]

    # Define receiver path and rotations
    with open(f'demo/config_demo/{novel_path_config}.json', 'r') as file:
        json_path = json.load(file)
    receiver_idx_list = json_path['receiver_idx_list']
    receiver_rotation_list = json_path['receiver_rotation_list']

    # Load grid points
    grid_points_receiver = load_room_grid(room, grid_distance=args.grid_distance)['grid_points']

    # Generate RIRs
    output_dir = f'{args.results_dir}/video_{channel_type}'
    os.makedirs(output_dir, exist_ok=True)
    ir_save_dir = f'{output_dir}/ir_save_{novel_path_config}_{channel_type}.pt'
    if os.path.exists(ir_save_dir):
        ir_output = torch.load(ir_save_dir)
    else:
        ir_output = generate_rir_combination(
            room, source_idx_list, grid_points_source,
            receiver_idx_list, receiver_rotation_list,
            grid_points_receiver, channel_type
        )
        torch.save(ir_output, ir_save_dir)
    ir1_list, ir2_list = ir_output

    # Prepare source audio
    if use_gt_location:
        source1_audio, _ = torchaudio.load(f'{args.results_dir}/results_drysound/dry1_estimated.wav')
        source2_audio, _ = torchaudio.load(f'{args.results_dir}/results_drysound/dry2_estimated.wav')
    else:
        source1_audio, _ = torchaudio.load(f'{args.results_dir}/results_drysound/detected/dry_{source_idx_list[0]}.wav')
        source2_audio, _ = torchaudio.load(f'{args.results_dir}/results_drysound/detected/dry_{source_idx_list[1]}.wav')

    source1_audio, source2_audio = clip_two(source1_audio, source2_audio)

    # Interpolate audio for moving receiver
    receiver_position = grid_points_receiver[receiver_idx_list]
    receiver_audio, receiver_audio_1, receiver_audio_2 = interpolate_moving_audio(source1_audio, source2_audio, ir1_list, ir2_list, receiver_position)

    # Save audio
    torchaudio.save(f'{output_dir}/moving_audio.wav', receiver_audio, sample_rate=sample_rate)
    torchaudio.save(f'{output_dir}/moving_audio_1.wav', receiver_audio_1, sample_rate=sample_rate)
    torchaudio.save(f'{output_dir}/moving_audio_2.wav', receiver_audio_2, sample_rate=sample_rate)

    # Cofigure scene to render images
    scene = configure_scene_from_metadata(metadata, use_placeholder_mesh=use_placeholder_mesh)

    # Interpolate RGB images
    audio_len_s = receiver_audio.shape[-1] / sample_rate
    video_len = int(sample_rate_video * audio_len_s)
    rgb_list = interpolate_rgb_images(scene, receiver_position, receiver_rotation_list, video_len)

    # Create video (no audio)
    fps = len(rgb_list) / audio_len_s
    writer = imageio.get_writer(f'{output_dir}/moving_video.mp4', fps=fps)
    for frame in rgb_list:
        writer.append_data(frame)
    writer.close()

    # Combine video and audio
    filenames_audio = ['moving_audio', 'moving_audio_1', 'moving_audio_2']
    filenames_nvas = ['nvas', 'nvas_source1', 'nvas_source2']
    for filename_audio, filename_nvas in zip(filenames_audio, filenames_nvas):
        command = [
            'ffmpeg', '-y',
            '-i', f'{output_dir}/moving_video.mp4',
            '-i', f'{output_dir}/{filename_audio}.wav',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            f'{output_dir}/{filename_nvas}.mp4'
        ]
        subprocess.run(command, check=True)

    # Render receiver view for reference
    receiver_position = grid_points_source[receiver_idx_list_original[0]].squeeze()
    receiver_rotation = -90
    receiver = Receiver(receiver_position, receiver_rotation)
    scene.update_receiver(receiver)
    rgb, depth = scene.render_envmap()
    plt.imsave(f'{output_dir}/rgb_receiver.png', rgb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_dir',
                         default='results/nvas3d_demo/default/demo/17DRP5sb8fy/0',
                         type=str,
                         help='results dir to generate demo video')

    parser.add_argument('--grid_distance',
                         default=0.5,
                         type=float,
                         help='distance between grid points to generate video')

    parser.add_argument('--sample_rate',
                        default=48000,
                        type=int,
                        help='Audio sample rate')

    parser.add_argument('--sample_rate_video',
                        default=30,
                        type=int,
                        help='Video sample rate')

    parser.add_argument('--use_gt_location',
                        default=False,
                        type=bool,
                        help='Use estimated audio from gt source location')

    parser.add_argument('--channel_type',
                        default='Binaural',
                        choices=['Binaural', 'Ambisonics'],
                        type=str,
                        help='Type of audio channel (choices: Binaural or Ambisonics).')

    parser.add_argument('--novel_path_config',
                        default='path1_17DRP5sb8fy',
                        type=str,
                        help='Novel receiver path')

    parser.add_argument('--use_placeholder_mesh',
                        default=True,
                        type=bool,
                        help='Use placeholder mesh at source and receiver locations')

    args = parser.parse_args()

    main(args)
