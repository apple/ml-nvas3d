#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import numpy as np
import glob
import torch
import matplotlib.pyplot as plt
from soundspaces_nvas3d.soundspaces_nvas3d import Receiver
from soundspaces_nvas3d.utils.aihabitat_utils import load_room_grid
from soundspaces_nvas3d.utils.ss_utils import create_scene
import argparse


def contains_png_files(dirname):
    return len(glob.glob(os.path.join(dirname, "*.png"))) > 0


def render_target_images(args):
    room = args.room
    grid_distance = args.grid_distance
    image_size = (192, 144)

    # Download soundspaces dataset if not exists
    grid_points = load_room_grid(room, grid_distance=grid_distance)['grid_points']
    grid_distance_str = str(grid_distance).replace(".", "_")
    if args.dirname is None:
        dirname = f'data/examples/target_image_mp3d/grid_{grid_distance_str}/{room}'
    else:
        dirname = args.dirname
    os.makedirs(dirname, exist_ok=True)

    # # Return if there is a png file already
    # if contains_png_files(dirname):
    #     return

    scene = create_scene(room, image_size=image_size)
    sample_rate = 48000
    position = [0.0, 0, 0]
    rotation = 0.0
    receiver = Receiver(position, rotation, sample_rate)

    # Render image from 1m distance
    dist = 1.0
    # south east north west
    position_offset_list = [[0.0, 0.0, dist], [dist, 0.0, 0.0], [0.0, 0.0, -dist], [-dist, 0.0, 0.0]]
    rotation_offset_list = [0.0, 90.0, 180.0, 270.0]

    source_class_list = ['female', 'male', 'guitar']
    # source_class_list = ['guitar']

    if args.source_idx_list is None:
        source_idx_list = range(grid_points.shape[0])
    else:
        source_idx_list = list(map(int, args.source_idx_list.split()))

    for source_idx in source_idx_list:
        # print(f'{room}: {source_idx}/{len(source_idx_list)}')
        source_position = grid_points[source_idx]

        rgb = []
        depth = []

        for position_offset, rotation_offset in zip(position_offset_list, rotation_offset_list):
            receiver.position = source_position + torch.tensor(position_offset)
            receiver.rotation = rotation_offset
            scene.update_receiver(receiver)
            rgb_, depth_ = scene.render_image()
            rgb.append(rgb_)
            depth.append(depth_)

        rgb = np.concatenate(rgb, axis=1)
        depth = np.concatenate(depth, axis=1)

        filename_rgb_png = f'{dirname}/rgb_{room}_{source_idx}.png'
        filename_depth_png = f'{dirname}/depth_{room}_{source_idx}.png'

        plt.imsave(filename_rgb_png, rgb)
        plt.imsave(filename_depth_png, depth)

        filename_rgb_npy = f'{dirname}/rgb_{room}_{source_idx}.npy'
        filename_depth_npy = f'{dirname}/depth_{room}_{source_idx}.npy'

        np.save(filename_rgb_npy, rgb)
        np.save(filename_depth_npy, depth)

        # Optional: Render with mesh. Requires mesh data for habitat under data/objects/{source.mesh}
        # source_female = Source(
        #     position=position,
        #     rotation=random.uniform(0, 360),
        #     dry_sound='',
        #     mesh='female',
        #     device=torch.device('cpu')
        # )

        # source_male = Source(
        #     position=position,
        #     rotation=random.uniform(0, 360),
        #     dry_sound='',
        #     mesh='male',
        #     device=torch.device('cpu')
        # )

        # source_guitar = Source(
        #     position=position,
        #     rotation=random.uniform(0, 360),
        #     dry_sound='',
        #     mesh='guitar',
        #     device=torch.device('cpu')
        # )

        # source_list = [source_female, source_male, source_guitar]
        # # source_list = [source_guitar]

        # for source_class, source in zip(source_class_list, source_list):
        #     scene.add_source_mesh = True
        #     scene.source_list = [None]
        #     source.position = source_position

        #     scene.update_source(source, 0)
        #     rgb = []
        #     depth = []

        #     for position_offset, rotation_offset in zip(position_offset_list, rotation_offset_list):
        #         receiver.position = source_position + torch.tensor(position_offset)
        #         receiver.rotation = rotation_offset
        #         scene.update_receiver(receiver)
        #         rgb_, depth_ = scene.render_image()
        #         rgb.append(rgb_)
        #         depth.append(depth_)

        #     scene.sim.get_rigid_object_manager().remove_all_objects()

        #     rgb = np.concatenate(rgb, axis=1)
        #     depth = np.concatenate(depth, axis=1)

        #     filename_rgb_png = f'{dirname}/rgb_{room}_{source_class}_{source_idx}.png'
        #     filename_depth_png = f'{dirname}/depth_{room}_{source_class}_{source_idx}.png'

        #     plt.imsave(filename_rgb_png, rgb)
        #     plt.imsave(filename_depth_png, depth)

        #     filename_rgb_npy = f'{dirname}/rgb_{room}_{source_class}_{source_idx}.npy'
        #     filename_depth_npy = f'{dirname}/depth_{room}_{source_class}_{source_idx}.npy'

        #     np.save(filename_rgb_npy, rgb)
        #     np.save(filename_depth_npy, depth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--room',
                         default='17DRP5sb8fy',
                         type=str,
                         help='mp3d room')

    parser.add_argument('--source_idx_list',
                         default=None,
                         type=str,
                         help='source_idx_list')

    parser.add_argument('--dirname',
                         default=None,
                         type=str,
                         help='dirname')

    parser.add_argument('--grid_distance',
                         default=1.0,
                         type=float,
                         help='distance between grid points')

    args = parser.parse_args()

    render_target_images(args)
