#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
# Portions of this code are derived from VIDA (CC-BY-NC).
# Original work available at: https://github.com/facebookresearch/learning-audio-visual-dereverberation/tree/main

import typing as T
import numpy as np

import torch
import torch.nn.functional as F

source_class_map = {
    'female': 0,  # speech: 22961
    'male': 1,
    'Bass': 2,  # 51678
    'Brass': 3,  # 7216
    'Chromatic Percussion': 4,  # 4154
    'Drums': 5,  # 47291
    'Guitar': 6,  # 75848
    'Organ': 7,  # 10438
    'Piano': 8,  # 59328
    'Pipe': 9,  # 8200
    'Reed': 10,  # 10039
    'Strings': 11,  # 3169
    'Strings (continued)': 12,  # 48429
    'Synth Lead': 13,  # 7053
    'Synth Pad': 14  # 11210
}

MP3D_SCENE_SPLITS = {
    'demo': ['17DRP5sb8fy'],
}


def get_key_from_value(my_dict, target_value):
    for key, value in my_dict.items():
        if value == target_value:
            return key
    return None


def parse_librispeech_metadata(filename: str) -> T.Dict:
    """ 
    Reads LibriSpeech metadata from a csv file and returns a dictionary.
    Each entry in the dictionary maps a reader_id (as integer) to its corresponding gender. 
    """

    import csv

    # Dictionary to store reader_id and corresponding gender
    librispeech_metadata = {}

    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='|')
        for row in reader:
            # Skip comment lines and header
            if row[0].startswith(';') or row[0].strip() == 'ID':
                continue
            reader_id = int(row[0])  # Convert string to integer
            sex = row[1].strip()  # Remove extra spaces
            librispeech_metadata[reader_id] = sex

    return librispeech_metadata


def parse_ir(filename_ir: str):
    """ 
    Extracts the room, source index and receiver index information from an IR filename. 
    The function assumes that these elements are present at the end of the filename, separated by underscores.
    """

    parts = filename_ir.split('_')
    room = parts[-3]
    source_idx = int(parts[-2])
    receiver_idx = int(parts[-1].split('.')[0])

    return room, source_idx, receiver_idx


def count_parameters(model):
    """ 
    Returns the number of trainable parameters in a given PyTorch model. 
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


####################
# From Changan's code
####################

def complex_norm(
        complex_tensor: torch.Tensor,
        power: float = 1.0
) -> torch.Tensor:
    """
    Compute the norm of complex tensor input.
    """

    # Replace by torch.norm once issue is fixed
    # https://github.com/pytorch/pytorch/issues/34279
    return complex_tensor.pow(2.).sum(-1).pow(0.5 * power)


def normalize(audio, norm='peak'):
    if norm == 'peak':
        peak = abs(audio).max()
        if peak != 0:
            return audio / peak
        else:
            return audio
    elif norm == 'rms':
        if torch.is_tensor(audio):
            audio = audio.numpy()
        audio_without_padding = np.trim_zeros(audio, trim='b')
        rms = np.sqrt(np.mean(np.square(audio_without_padding))) * 100
        if rms != 0:
            return audio / rms
        else:
            return audio
    else:
        raise NotImplementedError


def overlap_chunk(input, dimension, size, step, left_padding):
    """
    Input shape is [Frequency bins, Frame numbers]
    """
    input = F.pad(input, (left_padding, size), 'constant', 0)
    return input.unfold(dimension, size, step)
