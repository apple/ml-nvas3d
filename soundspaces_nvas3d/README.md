# SoundSpaces for NVAS3D
This guide provides a step-by-step process to set up and generate data using SoundSpaces for NVAS3D. You can also quickly start the installation process by running the `setup.sh` script included in this guide.

## Prerequisites for [SoundSpaces](https://github.com/facebookresearch/sound-spaces)
- Ubuntu 20.04 or a similar Linux distribution
- CUDA
- [Conda](https://docs.conda.io/en/latest/miniconda.html)


## Installation
Here we repeat the installation steps from [SoundSpaces Installation Guide](https://github.com/facebookresearch/sound-spaces/blob/main/INSTALLATION.md).

### 0. Update `PYTHONPATH`
Ensure your `PYTHONPATH` is updated:
```bash
cd ml-nvas3d # the root of the repository
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 1. Install SoundSpaces Dependencies
Install required dependencies for SoundSpaces:
```bash
apt-get update && apt-get upgrade -y && \
apt-get install -y --no-install-recommends libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
```

### 2. Update Conda
Ensure that your Conda installation is up to date:
```bash
conda update -n base -c defaults conda
```

### 3. Create and Activate Conda Environment
Create a new Conda environment named nvas3d with Python 3.7 and cmake 3.14.0:
```bash
conda create -n nvas3d python=3.7 cmake=3.14.0 -y && \
conda activate nvas3d
```

### 4. Install PyTorch
Install PyTorch, torchvision, torchaudio, and the CUDA toolkit. Replace the version accordingly with your CUDA version:
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

### 5. Install [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/main)
Install Habitat-Sim:
```bash
# step outside the repo root and git clone Habitat-Sim
cd .. && \ 
git clone https://github.com/facebookresearch/habitat-sim.git && \
cd habitat-sim && \
pip install -r requirements.txt && \
git checkout RLRAudioPropagationUpdate && \
python setup.py install --headless --audio --with-cuda
```
This guide is based on commit `30f4cc7`.

### 6. Install [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
Install Habitat-Lab:
```bash
# step outside the repo root and git clone Habitat-Lab
cd .. &&\
git clone https://github.com/facebookresearch/habitat-lab.git && \
cd habitat-lab && \
git checkout v0.2.2 && \
pip install -e . && \
sed -i '36 s/^/#/' habitat/tasks/rearrange/rearrange_sim.py # remove FetchRobot
```
This guide is based on commit `49f7c15`.


### 7. Install [SoundSpaces](https://github.com/facebookresearch/sound-spaces/tree/main)
Install SoundSpaces:
```bash
# step outside the repo root and git clone SoundSpaces
cd .. &&\ 
git clone https://github.com/facebookresearch/sound-spaces.git && \
cd sound-spaces && \
pip install -e .
```
This guide is based on commit `3768a50`.

### 8. Install Additional Packages
Install additional Python packages needed:
```bash
pip install scipy torchmetrics pyroomacoustics
```

## Quick Installation
To streamline the installation process, run the setup.sh script, which encapsulates all the steps listed above:
```bash
bash setup.sh
```

## Preparation of Matterport3D Room and Material

### 1. Download Example MP3D Room
Follow these steps to download the MP3D data in the correct directory:


(1) **Switch to habitat-sim directory**:
```bash
cd /path/to/habitat-sim
```

(2) **Run the dataset download script**:
```bash
python src_python/habitat_sim/utils/datasets_download.py --uids mp3d_example_scene
```

(3) **Copy the downloaded data to this repository**:
```bash
mkdir -p /path/to/ml-nvas3d/data/scene_datasets/ 
cp -r data/scene_datasets/mp3d_example /path/to/ml-nvas3d/data/scene_datasets/mp3d
```

After executing the above steps, ensure the existence of the `data/scene_datasets/mp3d/17DRP5sb8fy` directory. For additional rooms, you might want to consider downloading from [Matterport3D](https://niessner.github.io/Matterport/).

### 2. Download Material Configuration File
Download the material configuration file in the correct directory:

```bash
cd /path/to/ml-nvas3d && \
mkdir data/material && \
cd data/material && wget https://raw.githubusercontent.com/facebookresearch/rlr-audio-propagation/main/RLRAudioPropagationPkg/data/mp3d_material_config.json
```

> [!Note]
> Now, you are ready to run [Demo](../demo/README.md) using our pretrained model.
>
> To explore with more data, you can consider running [Training Data Generation](../nvas3d/utils/training_data_generation/README.md).


## Extended Usage: SoundSpaces for NVAS3D
For users interested in exploring with more data, follow the steps outlined below.

### Generate grid points in room:
To create grid points within the room, execute the following command:
```bash
python soundspaces_nvas3d/rir_generation/generate_grid.py --grid_distance 1.0
```
* Input directory: `data/scene_datasets/mp3d/`
* Output directory: `data/scene_datasets/metadata/mp3d/grid_{grid_distance}`

### Generate RIRs:
To generate room impulse responses for all grid point pairs, execute the following command:
```bash
python soundspaces_nvas3d/rir_generation/generate_rir.py --room 17DRP5sb8fy
```
* Input directory: `data/scene_datasets/mp3d/{room}`
* Output directory:`data/examples/rir_mp3d/grid_{grid_distance}/{room}`

### Minimal Example of RIR Generation Using SoundSpaces
We provide an mimimal example code for generating RIRs using our codebase. To generate sample RIRs, execute the following command:
```bash
python demo/sound_spaces/nvas3d/example_render_ir.py
```

---
Optionally, for those interested in training an audio-visual network, images can be rendered using the following methods:
### Render target images:
To render images centered at individual grid points, execute the following command:
```bash
python soundspaces_nvas3d/image_rendering/run_generate_target_image.py
```
* Input directory: `data/scene_datasets/mp3d/{room}`
* Output directory: `data/examples/target_image_mp3d/grid_{grid_distance}/{room}`

### Render environment maps:
To render environment maps from all grid points, execute the following command:
```bash
python soundspaces_nvas3d/image_rendering/run_generate_envmap.py
```
* Input directory: `data/scene_datasets/mp3d/{room}`
* Output directory: `data/examples/envmap_3d/grid_{grid_distance}/{room}`

