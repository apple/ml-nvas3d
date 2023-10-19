#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
# Portions of this code are derived from Habitat-Sim and Habitat-Lab (MIT licensed)
# Original work available at: https://aihabitat.org
#

import os
import numpy as np
import matplotlib.pyplot as plt
import typing as T

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
from scipy.interpolate import griddata

from habitat.utils.visualizations import maps


def load_room_grid(
    room: str,
    grid_distance: float
) -> T.Dict:
    """
    Load grid data for a specified room. If the grid data does not exist, it generates one.

    Args:
    - room:             Name of the room.
    - grid_distance:    The spacing between grid points.

    Returns:
    - A dictionary containing grid information for the specified room.
    """

    grid_distance_str = str(grid_distance).replace(".", "_")
    dirname_grid = f'data/scene_datasets/metadata/mp3d/grid_{grid_distance_str}'
    filename_grid = f'{dirname_grid}/grid_{room}.npy'
    if not os.path.exists(filename_grid):
        os.makedirs(dirname_grid, exist_ok=True)
        print(f'Computing grid_{room}...')
        from soundspaces_nvas3d.rir_generation.generate_grid import save_xy_grid_points
        grid_info = save_xy_grid_points(room, grid_distance, dirname_grid)

    # load grid
    grid_info = np.load(filename_grid, allow_pickle=True).item()

    return grid_info


def convert_points_to_topdown(
    pathfinder,
    points: T.List[T.Tuple[float, float, float]],
    meters_per_pixel: float
) -> T.List[T.Tuple[float, float]]:
    """
    Convert 3D points (x, z) to top-down view points (x, y).

    Args:
    - pathfinder:           The pathfinder object for conversion context.
    - points:               List of 3D points.
    - meters_per_pixel:     Scale for converting meters to pixels.

    Returns:
    - A list of converted points in top-down view.
    """

    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


def display_map(
    topdown_map,
    filename: T.Optional[str] = None,
    key_points: T.Optional[T.List[T.Tuple[float, float]]] = None,
    text_margin: int = 5,
    is_grid: bool = False
):
    """
    Display a top-down map. Optionally, plot key points on the map.

    Args:
    - topdown_map:      Topdown map
    - filename:         Filename to save the topdown map
    - key_points:       List of points to be highlighted on the map.
    - text_margin:      Margin for text labels, defaults to 5.
    - is_grid:          If True, considers the points as grid points.
    """

    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for i, point in enumerate(key_points):
            if is_grid:
                text_margin = 2
                plt.plot(point[0], point[1], marker=".", markersize=5, alpha=0.8, markerfacecolor='orange', markeredgecolor='none')
                plt.text(point[0] + text_margin, point[1] + text_margin, f'{i}', fontsize=6, color='orange')
            else:
                if i == 0:
                    # receiver
                    plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8, markerfacecolor='blue')
                    plt.text(point[0] + text_margin, point[1] + text_margin, 'receiver', fontsize=12, color='blue')
                else:
                    # sources
                    plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8, markerfacecolor='red')
                    plt.text(point[0] + text_margin, point[1] + text_margin, f'source {i}', fontsize=12, color='red')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show(block=False)


def save_town_map_grid(
    filename: str,
    pathfinder,
    grid_points: T.List[T.Tuple[float, float, float]],
    meters_per_pixel: float = 0.05
):
    """
    Generate a top-down view of a town map with grid points

    Args:
    - filename:             Filename to save town map image
    - pathfinder:           Pathfinder object used for contextual conversion.
    - grid_points:          List of 3D grid points.
    - meters_per_pixel:     Scale for converting meters to pixels. Defaults to 0.05.
    """

    assert pathfinder.is_loaded
    grid_points = np.array(grid_points)

    if len(grid_points) == 0:
        height = 0  # for empty grid_points
        xy_grid_points = None
    else:
        height = grid_points[0, 1]
        # Convert points to topdown
        xy_grid_points = convert_points_to_topdown(
            pathfinder, grid_points, meters_per_pixel
        )

    # Get topdown map
    top_down_map = maps.get_topdown_map(
        pathfinder, height=height, meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]

    # Save map
    display_map(top_down_map, key_points=xy_grid_points, filename=filename, is_grid=True)


def plot_heatmap(
    points: T.List[T.Tuple[float, float]],
    values: T.List[float],
    cmap: str = 'viridis',
    s: int = 100,
    alpha: float = 0.8,
    receiver_idx_list: T.Optional[T.List[int]] = None,
    source_idx_list: T.Optional[T.List[int]] = None,
    square_scale: int = 8
):
    """
    Plot a heatmap based on the given points and values.

    Args:
    - points:               List of 2D points.
    - values:               List of corresponding values for the points.
    - cmap:                 Colormap used for heatmap. Defaults to 'viridis'.
    - s:                    Marker size. Defaults to 100.
    - alpha:                Opacity of the markers. Defaults to 0.8.
    - receiver_idx_list:    Index list for receiver points.
    - source_idx_list:      Index list for source points.
    - square_scale:         Scaling factor for squares. Defaults to 8.
    """

    norm = Normalize(vmin=np.min(values), vmax=np.max(values))
    mapper = ScalarMappable(norm=norm, cmap=cmap)

    for i, (x, y) in enumerate(points):
        if receiver_idx_list is not None and i in receiver_idx_list:
            color = 'blue'  # Strong red color for points in receiver_idx_list
            marker = 's'
            s_scale = 1
        else:
            color = mapper.to_rgba(values[i])
            marker = 's'
            s_scale = square_scale

        plt.scatter(x, y, color=color, s=s * s_scale, alpha=alpha, marker=marker)

    for i, (x, y) in enumerate(points):
        if source_idx_list is not None and i in source_idx_list:
            color = 'red'
            marker = 'o'
            s_scale = 1

            plt.scatter(x, y, edgecolors=color, s=s_scale * s, facecolors='none', alpha=alpha, marker=marker)
    # plt.colorbar(mapper)


def plot_heatmap_interp(points, values, image, resolution=20, cmap='viridis', alpha=0.0):

    # Define the grid
    x_range = np.arange(0, image.shape[1], resolution)
    y_range = np.arange(0, image.shape[0], resolution)
    X, Y = np.meshgrid(x_range, y_range)

    # Interpolate the values
    Z = griddata(points, values, (X, Y), method='linear', fill_value=0)

    # Normalize the values and create a colormap
    norm = Normalize(vmin=np.min(values), vmax=np.max(values))
    mapper = ScalarMappable(norm=norm, cmap=cmap)

    # Plot the interpolated heatmap
    plt.imshow(mapper.to_rgba(Z), alpha=alpha, extent=(0, image.shape[1], 0, image.shape[0]), origin='upper')

    # Add a colorbar
    plt.colorbar(mapper)


def save_grid_heatmap(filename, pathfinder, grid_points, values, meters_per_pixel=0.05, receiver_idx_list=None, source_idx_list=None, best_idx_list=None):
    assert pathfinder.is_loaded

    grid_points = np.array(grid_points)

    if len(grid_points) == 0:
        height = 0  # for empty grid_points
        xy_grid_points = None
    else:
        height = grid_points[0, 1]
        # Convert points to topdown
        xy_grid_points = convert_points_to_topdown(
            pathfinder, grid_points, meters_per_pixel
        )

    # Get topdown map
    top_down_map = maps.get_topdown_map(
        pathfinder, height=height, meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]

    # Extract x and y coordinates
    points = np.array([point.flatten() for point in xy_grid_points])
    flipped_points = points.copy()
    flipped_points[:, 1] = top_down_map.shape[0] - points[:, 1]
    # x = points[:, 0]
    # y = points[:, 1]

    # # Create a grid for the heatmap
    # grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))

    # # Interpolate the values onto the grid
    # grid_values = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0)

    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(np.flipud(top_down_map))

    plot_heatmap(flipped_points, values, receiver_idx_list=receiver_idx_list, source_idx_list=source_idx_list)
    # plt.savefig('debug/scatter.png')

    if best_idx_list is not None:
        for i, (x, y) in enumerate(flipped_points):
            if i in best_idx_list:
                color = 'green'
                marker = 'x'
                s_scale = 1

                plt.scatter(x, y, color=color, s=s_scale * 100, alpha=1.0, marker=marker)

    plot_heatmap_interp(points, values, top_down_map)

    plt.savefig(filename)


def save_grid_heatmap_simple(
    filename: str,
    pathfinder,
    grid_points: T.List[T.Tuple[float, float, float]],
    values: T.List[float],
    meters_per_pixel: float = 0.05,
    receiver_idx_list: T.Optional[T.List[int]] = None,
    source_idx_list: T.Optional[T.List[int]] = None,
    best_idx_list: T.Optional[T.List[int]] = None
):
    """
    Generate a heatmap of the given grid points and values

    Args:
    - filename (str): Path where the image should be saved.
    - pathfinder: A pathfinder object used for contextual conversion.
    - grid_points (T.List): List of 3D grid points.
    - values (T.List): Corresponding values for each grid point.
    - meters_per_pixel (float, optional): Scale for converting meters to pixels. Defaults to 0.05.
    - receiver_idx_list (T.List, optional): Index list for receiver points.
    - source_idx_list (T.List, optional): Index list for source points.
    - best_idx_list (T.List, optional): Index list for the best points.
    """

    assert pathfinder.is_loaded
    grid_points = np.array(grid_points)

    if len(grid_points) == 0:
        height = 0  # for empty grid_points
        xy_grid_points = None
    else:
        height = grid_points[0, 1]
        # Convert points to topdown
        xy_grid_points = convert_points_to_topdown(
            pathfinder, grid_points, meters_per_pixel
        )

    # Get topdown map
    top_down_map = maps.get_topdown_map(
        pathfinder, height=height, meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]

    # Extract x and y coordinates
    points = np.array([point.flatten() for point in xy_grid_points])
    flipped_points = points.copy()
    flipped_points[:, 1] = top_down_map.shape[0] - points[:, 1]

    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")

    plot_heatmap(flipped_points, values, receiver_idx_list=receiver_idx_list, source_idx_list=source_idx_list, square_scale=0.25)

    if best_idx_list is not None:
        for i, (x, y) in enumerate(flipped_points):
            if i in best_idx_list:
                color = 'green'
                marker = 'x'
                s_scale = 1

                plt.scatter(x, y, color=color, s=s_scale * 100, alpha=1.0, marker=marker)

    plt.savefig(filename)


def save_grid_config(filename, pathfinder, grid_points, meters_per_pixel=0.05, receiver_idx_list=None, source_idx_list=None, no_query=False, query_idx_list=None):
    assert pathfinder.is_loaded

    grid_points = np.array(grid_points)
    if len(grid_points) == 0:
        height = 0  # for empty grid_points
        xy_grid_points = None
    else:
        height = grid_points[0, 1]
        # Convert points to topdown
        xy_grid_points = convert_points_to_topdown(
            pathfinder, grid_points, meters_per_pixel
        )

    # Get topdown map
    top_down_map = maps.get_topdown_map(
        pathfinder, height=height, meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]

    # Extract x and y coordinates
    points = np.array([point.flatten() for point in xy_grid_points])

    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(top_down_map)

    for i, (x, y) in enumerate(points):
        if source_idx_list is not None and i in source_idx_list:
            color = 'red'
            marker = 'o'
            s = 400
        elif receiver_idx_list is not None and i in receiver_idx_list:
            color = 'blue'  # Strong red color for points in receiver_idx_list
            marker = 's'
            s = 100
        else:
            if no_query:
                continue
            else:
                if query_idx_list is None or i in query_idx_list:
                    color = 'yellow'
                    marker = 'x'
                    s = 100
                else:
                    continue

        plt.scatter(x, y, color=color, s=s, alpha=1.0, marker=marker)

    plt.savefig(filename)


def display_sample(rgb_obs, semantic_obs: T.Optional[np.array] = np.array([]), depth_obs: T.Optional[np.array] = np.array([]), filename: T.Optional[str] = None):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")
        new_filename = os.path.splitext(filename)[0] + '_.' + os.path.splitext(filename)[1]
        semantic_img.save(new_filename)

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")

        ax.set_title(titles[i])
        plt.imshow(data)
    if filename is not None:
        plt.savefig(filename)
    plt.show(block=False)


def semantic_id_to_category_id(semantic_array, objects):
    unlabeled_id = 41

    category_array = np.zeros_like(semantic_array)
    for row in range(semantic_array.shape[0]):
        for col in range(semantic_array.shape[1]):
            semantic_id = semantic_array[row][col]
            if semantic_id >= len(objects):
                category_id = unlabeled_id
            else:
                category_id = objects[semantic_id].category.index()
            if category_id < 0 or category_id > unlabeled_id:  # usually category_id == -1
                category_id = unlabeled_id
            category_array[row][col] = category_id

    return category_array


def print_scene_recur(scene, limit_output=10):
    print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = args.display
    display = args.display
    do_make_video = args.make_video
else:
    show_video = False
    do_make_video = False
    display = False
