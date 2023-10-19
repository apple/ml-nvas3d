#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
# Script to clip and split Slakh2100 dataset: https://zenodo.org/record/4599666
#

import os
import yaml
import shutil
import numpy as np

import torchaudio
from torchaudio.transforms import Resample


# Set the silence threshold
THRESHOLD = 1e-2


def read_instruments(source_dir):
    """
    Read instrument metadata from source directory
    """

    inst_dict = {}  # dictionary to store instrument data
    # Loop through every file in the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith("metadata.yaml"):
                with open(os.path.join(root, file), 'r') as yaml_file:
                    metadata = yaml.safe_load(yaml_file)

                    # Add instrument name and class to dictionary
                    for stem, stem_data in metadata['stems'].items():
                        if stem_data['midi_program_name'] not in inst_dict.keys():
                            inst_dict[stem_data['midi_program_name']] = stem_data['inst_class']
    return inst_dict


def copy_files_based_on_metadata(source_dir, target_dir, query):
    """
    Copy files based on metadata
    """

    # Walk through each file in the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith("metadata.yaml"):
                with open(os.path.join(root, file), 'r') as yaml_file:
                    try:
                        metadata = yaml.safe_load(yaml_file)

                        # If instrument matches query, copy the associated flac file to target directory
                        for stem, stem_data in metadata['stems'].items():
                            if stem_data['midi_program_name'] == query:
                                source_flac_file = os.path.join(root.replace('metadata.yaml', ''), "stems", f"{stem}.flac")
                                target_flac_file = source_flac_file.replace(source_dir, target_dir)

                                os.makedirs(os.path.dirname(target_flac_file), exist_ok=True)

                                # Copy the .flac file
                                if os.path.exists(source_flac_file):
                                    shutil.copyfile(source_flac_file, target_flac_file)

                    except yaml.YAMLError as exc:
                        print(exc)


def save_clips(waveform, output_path, sample_rate, min_length, max_length):
    """
    Save clips from a given waveform
    """

    # Calculate clip lengths in samples
    min_length_samples = int(min_length * sample_rate)
    max_length_samples = int(max_length * sample_rate)

    # Get total number of samples in the waveform
    total_samples = waveform.shape[1]

    start = 0
    end = np.random.randint(min_length_samples, max_length_samples + 1)
    clip_number = 1

    # Keep creating clips until we've covered the whole waveform
    while end <= total_samples:
        # Slice the waveform to get the clip
        clip = waveform[:, start:end]

        # Check if the clip contains all zeros (is silent)
        if abs(clip).mean() > THRESHOLD:
            # Save the clip
            output_clip_path = f"{output_path.rsplit('.', 1)[0]}_clip_{clip_number}.flac"
            torchaudio.save(output_clip_path, clip, sample_rate)

            # Increment the clip number
            clip_number += 1

        # Update the start and end for the next clip
        start = end
        end = start + np.random.randint(min_length_samples, max_length_samples + 1)


def process_directory(input_dir, output_dir, sample_rate=48000, min_length=6, max_length=10):
    """
    Process a directory of audio files
    """

    # Walk through each file in the input directory
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".flac"):
                input_file_path = os.path.join(dirpath, filename)
                waveform_, sr = torchaudio.load(input_file_path)

                # Resample the audio
                upsample_transform = Resample(sr, sample_rate)
                waveform = upsample_transform(waveform_)

                relative_path = os.path.relpath(dirpath, input_dir)
                clip_output_dir = os.path.join(output_dir, relative_path)

                os.makedirs(clip_output_dir, exist_ok=True)

                output_file_path = os.path.join(clip_output_dir, filename)

                # set audio length
                split = dirpath.split('/')[-3]
                if split == 'test':
                    min_length = 20
                    max_length = 21
                else:
                    min_length = 6
                    max_length = 7
                # Save clips from the resampled audio
                save_clips(waveform, output_file_path, sample_rate, min_length, max_length)


# Read instruments
source_dir = 'data/source/slakh2100_flac_redux'
inst_dict = read_instruments(source_dir)
print(inst_dict)

# Copy query instruments
os.makedirs('data/MIDI', exist_ok=True)
os.makedirs('data/MIDI/full', exist_ok=True)

# Loop through each instrument in the dictionary
for query, inst_class in inst_dict.items():
    os.makedirs(f'data/MIDI/full/{inst_class}', exist_ok=True)
    target_dir = f'data/MIDI/full/{inst_class}/{query}'
    print(target_dir)

    # Copy each instrument file to the target directory
    for subdir in ['train', 'test', 'validation']:
        copy_files_based_on_metadata(os.path.join(source_dir, subdir), os.path.join(target_dir, subdir), query)
print('Copy done!')

# Clip the copied audio files
os.makedirs('data/MIDI/clip', exist_ok=True)
for query, inst_class in inst_dict.items():
    os.makedirs(f'data/MIDI/clip/{inst_class}', exist_ok=True)
    target_dir = f'data/MIDI/full/{inst_class}/{query}'
    clip_dir = f'data/MIDI/clip/{inst_class}/{query}'
    print(clip_dir)

    process_directory(target_dir, clip_dir)
print('Clip done!')
