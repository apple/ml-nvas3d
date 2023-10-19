#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import argparse
import itertools

from soundspaces_nvas3d.utils.ss_utils import render_rir_parallel
from soundspaces_nvas3d.utils.aihabitat_utils import load_room_grid


def generate_rir(args: argparse.Namespace) -> None:
    """
    Generate Room Impulse Response (RIR) based on given room and grid distance.
    """

    grid_distance_str = str(args.grid_distance).replace(".", "_")
    dirname = os.path.join(args.dirname, f'rir_mp3d/grid_{grid_distance_str}', args.room)
    os.makedirs(dirname, exist_ok=True)

    grid_data = load_room_grid(args.room, grid_distance=args.grid_distance)
    grid_points = grid_data['grid_points']
    num_points = len(grid_points)

    # Generate combinations of source and receiver indices
    pairs = list(itertools.product(range(num_points), repeat=2))
    source_indices, receiver_indices = zip(*pairs)

    room_list = [args.room] * len(source_indices)
    source_points = grid_points[list(source_indices)]
    receiver_points = grid_points[list(receiver_indices)]
    filename_list = [
        os.path.join(dirname, f'ir_{args.room}_{source_idx}_{receiver_idx}.wav')
        for source_idx, receiver_idx in zip(source_indices, receiver_indices)
    ]

    render_rir_parallel(room_list, source_points, receiver_points, filename_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Room Impulse Response for given parameters.")

    parser.add_argument('--room',
                        default='17DRP5sb8fy',
                        type=str,
                        help='MP3D room identifier')

    parser.add_argument('--grid_distance',
                        default=2.0,
                        type=float,
                        help='Distance between grid points in meters')

    parser.add_argument('--dirname',
                        default='data/examples',
                        type=str,
                        help='Directory to save generated RIRs')

    args = parser.parse_args()
    generate_rir(args)
