#!/bin/bash

# =========================================
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
# =========================================

# =============
# Configuration
# =============
# For data generation
room='17DRP5sb8fy'
dataset_dir='nvas3d_demo'
scene_config="scene1_17DRP5sb8fy"

# For dry-sound estimation
model='default'
results_dir="results/$dataset_dir/$model/demo/$room/0"

# For novel-view acoustic rendering
novel_path_config="path1_17DRP5sb8fy"


# =============
# Execution
# =============
# Data generation
python demo/generate_demo_data.py --room $room --dataset_dir $dataset_dir --scene_config $scene_config

# Dry-sound estimation using our model
python demo/test_demo.py --dataset_dir $dataset_dir --model $model

# Novel-view acoustic rendering
python demo/generate_demo_video.py --results_dir $results_dir --novel_path_config $novel_path_config

