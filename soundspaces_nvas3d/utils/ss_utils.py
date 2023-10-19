#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import sys
import random
import itertools
import typing as T
import multiprocessing
import matplotlib.pyplot as plt
from contextlib import contextmanager
from tqdm import tqdm

import torch
import torchaudio

from soundspaces_nvas3d.soundspaces_nvas3d import Receiver, Source, Scene
from soundspaces_nvas3d.utils.aihabitat_utils import save_grid_config, load_room_grid


@contextmanager
def suppress_stdout_and_stderr():
    """
    Suppress the logs from SoundSpaces.
    """

    original_stdout_fd = os.dup(sys.stdout.fileno())
    original_stderr_fd = os.dup(sys.stderr.fileno())
    devnull = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull, sys.stdout.fileno())
        os.dup2(devnull, sys.stderr.fileno())
        yield
    finally:
        os.dup2(original_stdout_fd, sys.stdout.fileno())
        os.dup2(original_stderr_fd, sys.stderr.fileno())
        os.close(devnull)


def create_scene(room: str,
                 receiver_position: T.Tuple[float, float, float] = [0.0, 0.0, 0.0],
                 sample_rate: float = 48000,
                 image_size: T.Tuple[int, int] = (512, 256),
                 include_visual_sensor: bool = True,
                 hfov: float = 90.0
                 ) -> Scene:
    """
    Create a soundspaces scene to render IR.
    """

    # Note: Make sure mp3d room is downloaded
    with suppress_stdout_and_stderr():
        # Create a receiver
        receiver = Receiver(
            position=receiver_position,
            rotation=0,
            sample_rate=sample_rate
        )

        scene = Scene(
            room,
            [None],  # placeholder for source class
            receiver=receiver,
            include_visual_sensor=include_visual_sensor,
            add_source_mesh=False,
            device=torch.device('cpu'),
            add_source=False,
            image_size=image_size,
            hfov=hfov
        )

    return scene


def render_ir(room: str,
              source_position: T.Tuple[float, float, float],
              receiver_position: T.Tuple[float, float, float],
              filename: str = None,
              receiver_rotation: float = None,
              sample_rate: float = 48000,
              use_default_material: bool = False,
              channel_type: str = 'Ambisonics',
              channel_order: int = 1
              ) -> torch.Tensor:
    """
    Render impulse response for a source and receiver pair in the mp3d room.
    """

    if receiver_rotation is None:
        receiver_rotation = 90

    # Create a receiver
    receiver = Receiver(
        position=receiver_position,
        rotation=receiver_rotation,
        sample_rate=sample_rate
    )

    # Create a source
    source = Source(
        position=source_position,
        rotation=0,
        dry_sound='',
        mesh='',
        device=torch.device('cpu')
    )

    scene = Scene(
        room,
        [None],  # placeholder for source class
        receiver=receiver,
        source_list=[source],
        include_visual_sensor=False,
        add_source_mesh=False,
        device=torch.device('cpu'),
        use_default_material=use_default_material,
        channel_type=channel_type,
        channel_order=channel_order
    )

    # Render IR
    scene.add_audio_sensor()
    # with suppress_stdout_and_stderr():
    ir = scene.render_ir(0)

    # Save file if dirname is given
    if filename is not None:
        torchaudio.save(filename, ir, sample_rate=sample_rate)
    else:
        return ir


def render_rir_parallel(room_list: T.List[str],
                        source_position_list: T.List[T.Tuple[float, float, float]],
                        receiver_position_list: T.List[T.Tuple[float, float, float]],
                        filename_list: T.List[str] = None,
                        receiver_rotation_list: T.List[float] = None,
                        batch_size: int = 64,
                        sample_rate: float = 48000,
                        use_default_material: bool = False,
                        channel_type: str = 'Ambisonics',
                        channel_order: int = 1
                        ) -> T.List[torch.Tensor]:
    """
    Run render_ir parallely for all elements of zip(source_position_list, receiver_position_list).
    """

    assert len(room_list) == len(source_position_list)
    assert len(source_position_list) == len(receiver_position_list)

    if filename_list is None:
        is_return = True
    else:
        is_return = False

    if receiver_rotation_list is None:
        receiver_rotation_list = [0] * len(receiver_position_list)

    # Note: Make sure all rooms are downloaded

    # Calculate the number of batches
    num_points = len(source_position_list)
    num_batches = (num_points + batch_size - 1) // batch_size

    # Use tqdm to display the progress bar
    progress_bar = tqdm(total=num_points)

    def update_progress(*_):
        progress_bar.update()

    ir_list = []
    # Process the tasks in batches
    for batch_idx in range(num_batches):
        # Calculate the start and end indices of the current batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_points)
        if is_return:
            batch = [(room_list[i], source_position_list[i], receiver_position_list[i], None, receiver_rotation_list[i]) for i in range(start_idx, end_idx)]
        else:
            batch = [(room_list[i], source_position_list[i], receiver_position_list[i], filename_list[i], receiver_rotation_list[i]) for i in range(start_idx, end_idx)]

        # Create a multiprocessing Pool for the current batch
        with multiprocessing.Pool() as pool:
            tasks = []
            for room, source_position, receiver_position, filename, receiver_rotation in batch:
                # Apply async mapping of process_ir function
                task = pool.apply_async(render_ir, args=(room, source_position, receiver_position, filename, receiver_rotation, sample_rate, use_default_material, channel_type, channel_order), callback=update_progress)
                tasks.append(task)

            # Wait for all tasks in the batch to complete and collect results
            for task in tasks:
                if is_return:
                    ir = task.get()  # Block until the result is ready
                    ir_list.append(ir)  # Append the result to the list
                else:
                    task.get()
    if is_return:
        return ir_list


def render_ir_parallel_room_idx(room: str,
                                source_idx_list: T.List[int],
                                receiver_idx_list: T.List[int],
                                filename: str = None,
                                grid_distance=1.0,
                                batch_size: int = 64,
                                sample_rate: float = 48000,
                                use_default_material: bool = False,
                                channel_type='Ambisonics'  # Binaural
                                ) -> T.List[torch.Tensor]:
    """
    Run render_ir parallely for all elements of all_pair(source_idx_list, receiver_idx_list)
    """

    grid_points = load_room_grid(room, grid_distance=grid_distance)['grid_points']

    source_idx_pair_list, receiver_idx_pair_list = all_pairs(source_idx_list, receiver_idx_list)  # only for filename
    receiver_points = grid_points[receiver_idx_list]
    source_points = grid_points[source_idx_list]

    source_points_pair, receiver_points_pair = all_pairs(source_points, receiver_points)

    room_list = [room] * len(source_points_pair)
    if filename is not None:
        filename_list = [f'{filename}_{room}_{source_idx}_{receiver_idx}.wav'
                         for source_idx, receiver_idx in zip(source_idx_pair_list, receiver_idx_pair_list)]
    else:
        filename_list = None

    # Render IR for grid points
    ir_list = render_rir_parallel(room_list,
                                  source_points_pair,
                                  receiver_points_pair,
                                  filename_list,
                                  batch_size=batch_size,
                                  sample_rate=sample_rate,
                                  use_default_material=use_default_material,
                                  channel_type=channel_type)

    return ir_list, source_idx_pair_list, receiver_idx_pair_list


def render_receiver_image(dirname: str,
                          room: str,
                          source_idx_list: T.List[int],
                          source_class_list: T.List[str],
                          receiver_idx_list: T.List[int],
                          filename: str = None,
                          grid_distance=1.0,
                          hfov=120,
                          image_size=(1024, 1024)
                          ):

    # load grid points
    grid_points = load_room_grid(room, grid_distance=grid_distance)['grid_points']

    receiver_points = grid_points[receiver_idx_list]
    source_points = grid_points[source_idx_list]
    # source_points_pair, receiver_points_pair = all_pairs(source_points, receiver_points)

    # initialize scene
    scene = create_scene(room, image_size=image_size, hfov=hfov)
    scene.add_source_mesh = True

    # initialize receiver
    sample_rate = 48000
    position = [0.0, 0, 0]
    rotation = 0.0
    receiver = Receiver(position, rotation, sample_rate)

    # set source
    source_list = []
    for source_idx, source_class in zip(source_idx_list, source_class_list):
        position = grid_points[source_idx]
        if source_class == 'male' or source_class == 'female':
            source = Source(
                position=position,
                rotation=random.uniform(0, 360),
                dry_sound='',
                mesh=source_class,
                device=torch.device('cpu')
            )
            source_list.append(source)
        else:
            source = Source(
                position=position,
                rotation=random.uniform(0, 360),
                dry_sound='',
                mesh='guitar',  # All instruments use guitar mesh
                device=torch.device('cpu')
            )
            source_list.append(source)

    # add mesh
    scene.source_list = [None] * len(source_idx_list)
    for id, source in enumerate(source_list):
        scene.update_source(source, id)

    # see source1 direction
    source_x = source_points[0][0]
    source_z = source_points[0][2]

    # render images
    rgb_list = []
    depth_list = []
    for receiver_idx in receiver_idx_list:
        receiver.position = grid_points[receiver_idx]
        # all receiver sees source 1 direction
        rotation_source1 = calculate_degree(source_x - receiver.position[0], source_z - receiver.position[2])
        receiver.rotation = rotation_source1
        scene.update_receiver(receiver)
        rgb, depth = scene.render_image()
        rgb_list.append(rgb)
        depth_list.append(depth)

    # for index debug
    # source_idx_pair_list, receiver_idx_pair_list = all_pairs(source_idx_list, receiver_idx_list)

    for i in range(len(receiver_idx_list)):
        plt.imsave(f'{dirname}/{i+1}.png', rgb_list[i])
        plt.imsave(f'{dirname}/{i+1}_depth.png', depth_list[i], cmap='gray')

    # return rgb_list, depth_list


def render_scene_config(filename: str,
                        room: str,
                        source_idx_list: T.List[int],
                        receiver_idx_list: T.List[int],
                        grid_distance=1.0,
                        no_query=False):

    # load grid points
    grid_points = load_room_grid(room, grid_distance=grid_distance)['grid_points']

    # config
    scene = create_scene(room)
    save_grid_config(filename, scene.sim.pathfinder, grid_points, receiver_idx_list=receiver_idx_list, source_idx_list=source_idx_list, no_query=no_query)


# Additional utility functions
def all_pairs(list1, list2):
    list_pair = list(itertools.product(list1, list2))

    list1_pair, list2_pair = zip(*list_pair)
    list1_pair = list(list1_pair)
    list2_pair = list(list2_pair)

    return list1_pair, list2_pair


def calculate_degree(x, y):
    radian = torch.atan2(y, x)
    degree = torch.rad2deg(radian)
    # Adjusting for the described mapping
    degree = (-degree - 90) % 360
    return degree
