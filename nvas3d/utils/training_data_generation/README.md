# NVAS3D Training Data Generation
This guid provides the process of generating training data for NVAS3D.

## Step-by-Step Instructions

### 1. Download Matterport3D Data
Download all rooms from [Matterport3D](https://niessner.github.io/Matterport/).

### 2. Download Source Audios
Download the following dataset for source audios and locate them at `data/source`
* [Slakh2100](https://zenodo.org/record/4599666)
* [LibriSpeech](https://www.openslr.org/12)

### 3. Preprocess Source Audios
#### 3.1. Clip, Split, and Upsample Slakh2100
To process Slakh2100 dataset (clipping, splitting, and upsampling to 48kHz), execute the following command:
```bash
python nvas3d/training_data_generation/script_slakh.py
```
The output will be located at `data/MIDI/clip/`.

#### 3.2. Upsample LibriSpeech
To upsample LibriSpeech dataset to 48kHz, execute the following command:
```bash
python nvas3d/training_data_generation/upsample_librispeech.py
```
The output will be located at `data/source/LibriSpeech48k`, and move it to `data/MIDI/clip/speech/LibriSpeech48k`.

### 4. Generate Metadata for Microphone Configuration
To generate square-shaped microphone configuration metadata, execute the following command:
```bash
python nvas3d/training_data_generation/generate_metadata_square.py
```
The output metadata will be located at `data/nvas3d_square/`

### 5. Generate Training Data
Finally, to generate the training data for NVAS3D, execute the following command:
```bash
python nvas3d/training_data_generation/generate_training_data.py
```
The generated data will be located at `data/nvas3d_square_all_all`.

## Acknowledgements
* [LibriSpeech](https://www.openslr.org/12) is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

* [Slakh2100](http://www.slakh.com) is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

* [Matterport3D](https://niessner.github.io/Matterport/): Matterport3D-based task datasets and trained models are distributed with the [Matterport3D Terms of Use](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and are licensed under [CC BY-NC-SA 3.0 US](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).


