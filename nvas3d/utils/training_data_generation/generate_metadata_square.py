#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import json
import random
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

from soundspaces_nvas3d.utils.aihabitat_utils import load_room_grid
from nvas3d.utils.utils import MP3D_SCENE_SPLITS


def find_squares(points, grid_distance, tolerance=1e-5):
    squares = []
    for i, point1 in enumerate(points):
        x1, y1, z1 = point1
        for j, point2 in enumerate(points):
            if i != j:
                x2, y2, z2 = point2
                if abs(x2 - (x1 + grid_distance)) < tolerance and abs(z2 - z1) < tolerance:
                    for k, point3 in enumerate(points):
                        if k != i and k != j:
                            x3, y3, z3 = point3
                            if abs(x3 - x1) < tolerance and abs(z3 - (z1 + grid_distance)) < tolerance:
                                for l, point4 in enumerate(points):
                                    if l != i and l != j and l != k:
                                        x4, y4, z4 = point4
                                        if abs(x4 - (x1 + grid_distance)) < tolerance and abs(z4 - (z1 + grid_distance)) < tolerance:
                                            squares.append((i, j, l, k))
    return squares


def plot_points_and_squares(filename, points, squares):
    fig, ax = plt.subplots()

    # plot all points
    ax.scatter(*zip(*[(x, z) for x, _, z in points]), c='blue')

    # plot squares
    for square_indices in squares:
        square_points = [points[i] for i in square_indices]
        square_points.append(square_points[0])  # Add the first point again to close the square
        ax.plot(*zip(*[(x, z) for x, _, z in square_points]), c='red')

    ax.set_aspect('equal', 'box')  # Ensure the aspect ratio is equal
    plt.savefig(filename)
    plt.show()


def main(args):
    dataset_name = args.dataset_name
    num_pairs_per_room = args.num_pairs_per_room
    random.seed(0)

    grid_distance = args.grid_distance
    grid_distance_str = str(grid_distance).replace(".", "_")

    os.makedirs(f'data/{dataset_name}/metadata/grid_{grid_distance_str}', exist_ok=True)

    filesize_valid_total = 0.0
    filesize_invalid_total = 0.0
    num_valid_total = 0.0
    num_invalid_total = 0.0
    mean_firstnz_total = 0.0
    mean_rt60_total = 0.0
    mean_maximum_total = 0.0
    mean_mean_total = 0.0
    mean_num_points_total = 0.0
    mean_room_size_total = 0.0

    count_total = 0

    is_debug = False
    # Read jsons
    for split in ['demo']:  # ['train', 'val', 'test', 'demo']:

        filesize_valid_total_ = 0.0
        filesize_invalid_total_ = 0.0
        num_valid_total_ = 0.0
        num_invalid_total_ = 0.0
        mean_firstnz_total_ = 0.0
        mean_rt60_total_ = 0.0
        mean_maximum_total_ = 0.0
        mean_mean_total_ = 0.0
        mean_num_points_total_ = 0.0
        mean_room_size_total_ = 0.0
        count_split = 0

        for i_room, room in enumerate(MP3D_SCENE_SPLITS[split]):
            if is_debug:
                print(f'room: {room} in {split} ({i_room}/{len(MP3D_SCENE_SPLITS[split])})')
            filename = f'data/metadata_grid/grid_{grid_distance_str}/{room}.json'
            with open(filename, 'r') as file:
                metadata = json.load(file)

            filesize_valid_total += metadata['filesize_valid_in_mb']
            filesize_invalid_total += metadata['filesize_invalid_in_mb']
            num_valid_total += metadata['num_valid']
            num_invalid_total += metadata['num_invalid']
            mean_firstnz_total = metadata['mean_firstnz']
            mean_rt60_total = metadata['mean_rt60']
            mean_maximum_total = metadata['mean_maximum']
            mean_mean_total = metadata['mean_mean']
            mean_num_points_total = metadata['num_points']
            mean_room_size_total = metadata['room_size']

            filesize_valid_total_ += metadata['filesize_valid_in_mb']
            filesize_invalid_total_ += metadata['filesize_invalid_in_mb']
            num_valid_total_ += metadata['num_valid']
            num_invalid_total_ += metadata['num_invalid']
            mean_firstnz_total_ = metadata['mean_firstnz']
            mean_rt60_total_ = metadata['mean_rt60']
            mean_maximum_total_ = metadata['mean_maximum']
            mean_mean_total_ = metadata['mean_mean']
            mean_num_points_total_ = metadata['num_points']
            mean_room_size_total_ = metadata['room_size']

            count_split += 1
            count_total += 1

            # find valid
            points = load_room_grid(room, grid_distance)['grid_points']
            squares = find_squares(points, grid_distance)
            # filename = f'data/metadata_grid/grid_{grid_distance_str}/{room}_square.png'
            # plot_points_and_squares(filename, points, squares)

            # Initialize defaultdicts to count source_idx and receiver_idx occurrences
            source_idx_counts = defaultdict(set)
            receiver_idx_counts = defaultdict(set)

            # Iterate over keys in valid metadata
            for key in metadata.keys():
                if key.startswith(room):
                    # Extract source_idx and receiver_idx from key
                    room, source_idx, receiver_idx = key.split("_")

                    if metadata[key]['Is Valid'] == 'True':
                        # Add receiver_idx to the set associated with source_idx
                        # and vice versa
                        source_idx_counts[source_idx].add(receiver_idx)
                        receiver_idx_counts[receiver_idx].add(source_idx)

            # Initialize the empty dictionary where each value is a set
            square_to_source_idxs = defaultdict(set)

            # Iterate over each source_idx and its corresponding valid indices
            for source_idx, valid_indices in source_idx_counts.items():
                # Iterate over each square
                for square in squares:
                    # Check if all indices of the current square are valid for the current source_idx
                    if all(str(idx) in valid_indices for idx in square):
                        # Add the source_idx to the square's set of valid source_idxs
                        square_to_source_idxs[square].add(int(source_idx))

            # Filter out squares that have less than 3 valid source_idxs (2 for positive, 1 for negative)
            three_source_idx_squares = {square: source_idxs for square, source_idxs in square_to_source_idxs.items() if len(source_idxs) >= 3}

            # Initialize the list of pairs
            pairs = []

            # Iterate over the dictionary to build pairs
            for square, source_idxs in three_source_idx_squares.items():
                # Generate all combinations of 3 source indices
                for source_idx_pair in combinations(source_idxs, 3):
                    # Avoid adding pairs where any source_idx is in square (receiver indices)
                    if not any(source_idx in square for source_idx in source_idx_pair):
                        # Find novel receiver that is valid to source 1 and source 2
                        common_receiver = source_idx_counts[str(source_idx_pair[0])].intersection(source_idx_counts[str(source_idx_pair[1])])
                        common_receiver = common_receiver - set(square) - set([source_idx_pair[0]]) - set([source_idx_pair[1]]) - set(square)
                        if common_receiver:
                            novel = int(random.choice(list(common_receiver)))

                            # Add the pair to the list
                            pairs.append((source_idx_pair, square, novel))

            # If there are less than num_pairs_per_room
            if len(pairs) < num_pairs_per_room:
                selected_pairs = pairs
            else:
                # Randomly select num_pairs_per_room
                selected_pairs = random.sample(pairs, num_pairs_per_room)

            # Save the lists to JSON files
            total_dict = {}
            total_dict['squares'] = squares
            total_dict['selected_pairs'] = selected_pairs
            # total_dict['square_to_source_idxs_keys'] = list(square_to_source_idxs.keys())
            # total_dict['square_to_source_idxs_values'] = list(square_to_source_idxs.values())

            filename = f'data/{dataset_name}/metadata/grid_{grid_distance_str}/{room}_square.json'
            with open(filename, 'w') as file:
                json.dump(total_dict, file)

        if is_debug:
            print(split)
            print(filesize_valid_total_ / 1024)  # GB
            print(filesize_invalid_total_ / 1024)  # GB
            print(num_valid_total_)
            print(num_invalid_total_)
            print(mean_firstnz_total_ / count_split)
            print(mean_rt60_total_ / count_split)
            print(mean_maximum_total_ / count_split)
            print(mean_mean_total_ / count_split)
            print(mean_num_points_total_ / count_split)
            print(mean_room_size_total_ / count_split)

    if is_debug:
        print(filesize_valid_total / 1024)  # GB
        print(filesize_invalid_total / 1024)  # GB
        print(num_valid_total)
        print(num_invalid_total)
        print(mean_firstnz_total / count_total)
        print(mean_rt60_total / count_total)
        print(mean_maximum_total / count_total)
        print(mean_mean_total / count_total)
        print(mean_num_points_total / count_total)
        print(mean_room_size_total / count_total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate metadata for square-shape microphone array.")

    parser.add_argument('--grid_distance',
                        default=1.0,
                        type=float,
                        help='Distance between grid points in meters')

    parser.add_argument('--num_pairs_per_room',
                        default=1000,
                        type=int,
                        help='Number of pairs per room to generate')

    parser.add_argument('--dataset_name',
                        default='nvas3d_square',
                        type=str,
                        help='Name of the dataset')

    args = parser.parse_args()
    main(args)
