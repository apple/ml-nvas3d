#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import sys
from soundspaces_nvas3d.utils.ss_utils import render_scene_config, render_receiver_image, create_scene
from soundspaces_nvas3d.utils.aihabitat_utils import save_grid_heatmap


def execute_scene(args):
    filename, room, source_idx_list, all_receiver_idx_list, grid_distance = args
    render_scene_config(filename, room, eval(source_idx_list), eval(all_receiver_idx_list), eval(grid_distance), no_query=False)


def execute_receiver(args):
    dirname, room, source_idx_list, source_class_list, all_receiver_idx_list = args
    render_receiver_image(dirname, room, eval(source_idx_list), eval(source_class_list), eval(all_receiver_idx_list))


def execute_heatmap(args):
    filename, room, grid_points, prediction_list, receiver_idx_list, grid_distance = args
    scene = create_scene(room)
    save_grid_heatmap(filename, scene.sim.pathfinder, eval(grid_points), eval(prediction_list), receiver_idx_list=eval(receiver_idx_list))


# Dictionary to map function names to actual function calls
functions = {
    "scene": execute_scene,
    "receiver": execute_receiver,
    "heatmap": execute_heatmap
}

function_name = sys.argv[1]

# Execute the corresponding function
if function_name in functions:
    functions[function_name](sys.argv[2:])
else:
    print(f"Error: Function {function_name} not found!")
