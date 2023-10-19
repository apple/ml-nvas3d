#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import subprocess

# RENDER TARGET IMAGE
room_list = [
    '17DRP5sb8fy'
]

for i, room in enumerate(room_list):
    print(f'mp3d target image rendering: {room}, {i+1}/{len(room_list)}')
    subprocess.run(['python', 'soundspaces_nvas3d/image_rendering/generate_target_image.py', '--room', room])
