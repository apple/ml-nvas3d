#!/bin/bash

# =========================================
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
# =========================================
#
# This script automates the installation process described in the soundspaces_nvas3d/README.
# Please refer to soundspaces_nvas3d/README.md for detailed instructions and explanations.

# Update PYTHONPATH
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)" >> ~/.bashrc && source ~/.bashrc


# Install dependencies for Soundspaces
apt-get update && apt-get upgrade -y

apt-get install -y --no-install-recommends \
        libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev

# Update conda
conda update -n base -c defaults conda

# Create conda environment
conda create -n ml-nvas3d python=3.7 cmake=3.14.0 -y
conda activate ml-nvas3d

# Install torch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# Install habitat-sim
cd ..
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
git checkout RLRAudioPropagationUpdate
python setup.py install --headless --audio --with-cuda 

# Install habitat-lab
cd ..
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.2.2
pip install -e .
sed -i '36 s/^/#/' habitat/tasks/rearrange/rearrange_sim.py # remove FetchRobot

# Install soundspaces
cd ..
git clone https://github.com/facebookresearch/sound-spaces.git
cd sound-spaces
pip install -e .

# Change directory
cd ml-nvas3d