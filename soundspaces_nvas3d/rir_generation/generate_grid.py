#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import argparse

import numpy as np
import torch
import typing as T

from soundspaces_nvas3d.soundspaces_nvas3d import Scene

# Set rooms for which grids need to be generated
ROOM_LIST = [
    '17DRP5sb8fy'
]


def save_xy_grid_points(room: str,
                        grid_distance: float,
                        dirname: str
                        ) -> T.Dict[str, T.Any]:
    """
    Save xy grid points given a mp3d room
    """

    filename_npy = f'{dirname}/grid_{room}.npy'
    filename_png = f'{dirname}/grid_{room}.png'

    scene = Scene(
        room,
        [None],  # placeholder for source class
        include_visual_sensor=False,
        add_source_mesh=False,
        device=torch.device('cpu')
    )
    grid_points = scene.generate_xy_grid_points(grid_distance, filename_png=filename_png)
    room_size = scene.sim.pathfinder.navigable_area

    grid_info = dict(
        grid_points=grid_points,
        room_size=room_size,
        grid_distance=grid_distance,
    )
    np.save(filename_npy, grid_info)

    return grid_info


def calculate_statistics(data_dict):
    values = list(data_dict.values())
    average = np.mean(values)
    minimum = np.min(values)
    maximum = np.max(values)
    return average, minimum, maximum


def main(args):
    # Set grid distance
    grid_distance = args.grid_distance
    grid_distance_str = str(grid_distance).replace(".", "_")

    # Generate grid points
    dirname = f'data/scene_datasets/metadata/mp3d/grid_{grid_distance_str}'
    os.makedirs(dirname, exist_ok=True)

    # Define lists to store data
    num_points_dict = {}
    room_size_dict = {}

    for room in ROOM_LIST:
        # Note: mp3d room should be under data/scene_datasets/mp3d/{room}
        grid_info = save_xy_grid_points(room, grid_distance, dirname)

        # Append data to lists
        num_points_dict[room] = grid_info['grid_points'].shape[0]
        room_size_dict[room] = grid_info['room_size']

    # Calculate statistics
    num_points_average, num_points_min, num_points_max = calculate_statistics(num_points_dict)
    room_size_average, room_size_min, room_size_max = calculate_statistics(room_size_dict)

    # Save statistics to txt file
    filename_satistics = f'data/scene_datasets/metadata/mp3d/grid_{grid_distance_str}/statistics.txt'
    with open(filename_satistics, 'w') as file:
        file.write('[Number of points statistics]:\n')
        file.write(f'Average: {num_points_average:.2f}\n')
        file.write(f'Minimum: {num_points_min}\n')
        file.write(f'Maximum: {num_points_max}\n')
        file.write('\n')
        file.write('[Room size statistics]:\n')
        file.write(f'Average: {room_size_average:.2f}\n')
        file.write(f'Minimum: {room_size_min:.2f}\n')
        file.write(f'Maximum: {room_size_max:.2f}\n')

        file.write('\n')
        file.write('[Room-wise statistics]:\n')
        for room in ROOM_LIST:
            file.write(f'{room} - Num Points: {num_points_dict[room]}, \t Room Size: {room_size_dict[room]:.2f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--grid_distance',
                         default=1.0,
                         type=float,
                         help='distance between grid points')

    args = parser.parse_args()

    main(args)
