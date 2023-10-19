#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from soundspaces_nvas3d.utils.ss_utils import create_scene

"""
Notes:
- MP3D room should be located at: data/scene_datasets/mp3d/{room}
- Grid data should be present at: data/scene_datasets/metadata/mp3d/grid_1_0/grid_{room}.npy
  (Refer to: rir_generation/generate_grid.py for grid generation)
"""

room_list = [
    '17DRP5sb8fy'
]


def main(args):
    room = args.room
    grid_distance = args.grid_distance

    print(room)
    image_size = (192, 144)

    # Load grid
    grid_distance_str = str(grid_distance).replace(".", "_")
    dirname_grid = f'data/scene_datasets/metadata/mp3d/grid_{grid_distance_str}'
    filename_grid = f'{dirname_grid}/grid_{room}.npy'
    grid_info = np.load(filename_grid, allow_pickle=True).item()

    grid_points = grid_info['grid_points']
    dirname = f'data/examples/envmap_mp3d/grid_{grid_distance_str}/{room}'
    os.makedirs(dirname, exist_ok=True)

    scene = create_scene(room, image_size=image_size)

    for receiver_idx in range(grid_points.shape[0]):
        receiver_position = grid_points[receiver_idx]
        scene.update_receiver_position(receiver_position)
        rgb, depth = scene.render_envmap()

        filename_rgb = f'{dirname}/envmap_rgb_{room}_{receiver_idx}.png'
        filename_depth = f'{dirname}/envmap_depth_{room}_{receiver_idx}.png'

        plt.imsave(filename_rgb, rgb)
        plt.imsave(filename_depth, depth)

        filename_rgb = f'{dirname}/envmap_rgb_{room}_{receiver_idx}.npy'
        filename_depth = f'{dirname}/envmap_depth_{room}_{receiver_idx}.npy'

        np.save(filename_rgb, rgb)
        np.save(filename_depth, depth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--room',
                         default='17DRP5sb8fy',
                         type=str,
                         help='mp3d room')

    parser.add_argument('--grid_distance',
                         default=1.0,
                         type=float,
                         help='distance between grid points')

    args = parser.parse_args()

    main(args)
