#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import itertools

from soundspaces_nvas3d.utils.ss_utils import render_rir_parallel
from soundspaces_nvas3d.utils.aihabitat_utils import load_room_grid

# Configuration
grid_distance = 2.0
room = '17DRP5sb8fy'

# Load grid points for the specified room
grid_data = load_room_grid(room, grid_distance=grid_distance)
grid_points = grid_data['grid_points']

# Generate all possible combinations of source and receiver indices
num_points = len(grid_points)
pairs = list(itertools.product(range(num_points), repeat=2))
idx_source, idx_receiver = zip(*pairs)

# Extract corresponding source and receiver points
source_points = grid_points[list(idx_source)]
receiver_points = grid_points[list(idx_receiver)]

# Create a room list for the IR generator
room_list = [room] * len(source_points)

# Generate Room Impulse Responses (IRs) without saving as WAV
ir_list = render_rir_parallel(room_list, source_points, receiver_points)
