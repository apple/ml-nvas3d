#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import glob
import random
import numpy as np
import typing as T
from scipy.signal import fftconvolve

import torch
import torchaudio
import torch.nn.functional as F

MIN_LENGTH_AUDIO = 5 * 48000


def load_ir_source_receiver(
    ir_dir: str,
    room: str,
    source_idx: int,
    receiver_idx_list: T.List[int],
    ir_length: int
) -> T.List[torch.Tensor]:
    """
    Load impulse responses for specific source and receivers in a room.

    Args:
    - ir_dir:               Directory containing impulse response files.
    - room:                 Name of the room.
    - source_idx:           Index of the source.
    - receiver_idx_list:    List of receiver indices.
    - ir_length:            Length of the impulse response to be loaded.

    Returns:
    - List of loaded impulse responses (first channel only).
    """

    ir_list = []
    for receiver_idx in receiver_idx_list:
        filename_ir = f'{ir_dir}/{room}/ir_{room}_{source_idx}_{receiver_idx}.wav'
        ir, _ = torchaudio.load(filename_ir)
        if ir[0].shape[0] > ir_length:
            ir0 = ir[0][:ir_length]
        else:
            ir0 = F.pad(ir[0], (0, ir_length - ir[0].shape[0]))
        ir_list.append(ir0)

    return ir_list


def load_ir_source_receiver_allchannel(
    ir_dir: str,
    room: str,
    source_idx: int,
    receiver_idx_list: T.List[int],
    ir_length: int
) -> T.List[torch.Tensor]:
    """
    Load impulse responses for all channels for specific source and receivers in a room.

    Args:
    - ir_dir:               Directory containing impulse response files.
    - room:                 Name of the room.
    - source_idx:           Index of the source.
    - receiver_idx_list:    List of receiver indices.
    - ir_length:            Length of the impulse response to be loaded.

    Returns:
    - List of loaded impulse responses for all channels.
    """

    ir_list = []
    for receiver_idx in receiver_idx_list:
        filename_ir = f'{ir_dir}/{room}/ir_{room}_{source_idx}_{receiver_idx}.wav'
        ir, _ = torchaudio.load(filename_ir)
        if ir.shape[-1] > ir_length:
            ir = ir[..., :ir_length]
        else:
            ir = F.pad(ir, (0, ir_length - ir[0].shape[0]))
        ir_list.append(ir)

    return ir_list


def save_audio_list(
    filename: str,
    audio_list: T.List[torch.Tensor],
    sample_rate: int,
    audio_format: str
):
    """
    Save a list of audio tensors to files.

    Args:
    - filename:     Filename to save audio.
    - audio_list:   List of audio tensors to save.
    - sample_rate:  Sample rate of audio.
    - audio_format: File format to save audio.
    """

    for idx_audio, audio in enumerate(audio_list):
        torchaudio.save(f'{filename}_{idx_audio+1}.{audio_format}', audio.unsqueeze(0), sample_rate)


def clip_source(
    source1_audio: torch.Tensor,
    source2_audio: torch.Tensor,
    len_clip: int
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    """
    Clip source audio tensors for faster convolution.

    Args:
    - source1_audio:    First source audio tensor.
    - source2_audio:    Second source audio tensor.
    - len_clip:         Desired length of the output audio tensors.

    Returns:
    - Clipped source1_audio and source2_audio.
    """

    # pad audio
    if len_clip > source1_audio.shape[0]:
        source1_audio = F.pad(source1_audio, (0, len_clip - source1_audio.shape[0]))
        source1_audio = F.pad(source1_audio, (0, max(0, len_clip - source1_audio.shape[0])))
        source2_audio = F.pad(source2_audio, (0, max(0, len_clip - source2_audio.shape[0])))

    # clip
    start_index = np.random.randint(0, source1_audio.shape[0] - len_clip) \
        if source1_audio.shape[0] != len_clip else 0
    source1_audio_clipped = source1_audio[start_index: start_index + len_clip]
    source2_audio_clipped = source2_audio[start_index: start_index + len_clip]

    return source1_audio_clipped, source2_audio_clipped


def compute_reverb(
    source_audio: torch.Tensor,
    ir_list: T.List[torch.Tensor],
    padding: str = 'valid'
) -> T.List[torch.Tensor]:
    """
    Compute reverberated audio signals by convolving source audio with impulse responses.

    Args:
    - source_audio:     Source audio signal (dry) to be reverberated.
    - ir_list:          List of impulse responses for reverberation.
    - padding:          Padding mode for convolution ('valid' or 'full').

    Returns:
    - A list of reverberated audio signals.
    """

    reverb_list = []
    for ir in ir_list:
        reverb = fftconvolve(source_audio, ir, padding)
        reverb_list.append(torch.from_numpy(reverb))

    return reverb_list


####################
# Sampling source audio to generate data
####################

def sample_speech(files_librispeech, librispeech_metadata):
    source_speech = torch.zeros(1)  # Initialize with a tensor of zeros
    while torch.all(source_speech == 0) or source_speech.shape[-1] < MIN_LENGTH_AUDIO:  # Continue until a non-zero tensor is found
        filename_source = random.choice(files_librispeech)
        speaker_id = int(filename_source.split('/')[6])
        speaker_gender = librispeech_metadata[speaker_id]
        if speaker_gender == 'M':
            source_class = 'male'
        else:
            source_class = 'female'
        source_speech, _ = torchaudio.load(filename_source)
        source_speech = source_speech.reshape(-1)

    return source_speech, source_class


def sample_nonspeech(all_instruments_dir):
    class_dir = random.choice(all_instruments_dir)

    # Ensure that the class is not 'Speech'
    while 'Speech' in class_dir:
        class_dir = random.choice(all_instruments_dir)

    files_source = glob.glob(class_dir + '/**/*.flac', recursive=True)

    source_audio = torch.zeros(1)  # Initialize with a tensor of zeros
    while torch.all(source_audio == 0) or source_audio.shape[-1] < MIN_LENGTH_AUDIO:  # Continue until a non-zero tensor is found
        filename_source = random.choice(files_source)
        source_class = class_dir.split('/')[3]
        source_audio, _ = torchaudio.load(filename_source)
        source_audio = source_audio.reshape(-1)

    return source_audio, source_class


def sample_acoustic_guitar(all_instruments_dir):
    guitar_dir = [dirname for dirname in all_instruments_dir if dirname.split('/')[4] == 'Acoustic Guitar (steel)']

    class_dir = random.choice(guitar_dir)

    files_source = glob.glob(class_dir + '/**/*.flac', recursive=True)

    source_audio = torch.zeros(1)  # Initialize with a tensor of zeros
    while torch.all(source_audio == 0) or source_audio.shape[-1] < MIN_LENGTH_AUDIO:  # Continue until a non-zero tensor is found
        filename_source = random.choice(files_source)
        source_class = 'guitar'
        source_audio, _ = torchaudio.load(filename_source)
        source_audio = source_audio.reshape(-1)

    return source_audio, source_class


def sample_instrument(all_instruments_dir, librispeech_metadata, classname):
    guitar_dir = [dirname for dirname in all_instruments_dir if dirname.split('/')[3] == classname]  # e.g., Guitar

    class_dir = random.choice(guitar_dir)

    files_source = glob.glob(class_dir + '/**/*.flac', recursive=True)

    source_audio = torch.zeros(1)  # Initialize with a tensor of zeros
    while torch.all(source_audio == 0) or source_audio.shape[-1] < MIN_LENGTH_AUDIO:  # Continue until a non-zero tensor is found
        filename_source = random.choice(files_source)
        source_class = 'guitar'
        source_audio, _ = torchaudio.load(filename_source)
        source_audio = source_audio.reshape(-1)

    return source_audio, source_class


def sample_all(all_instruments_dir, librispeech_metadata):
    class_dir = random.choice(all_instruments_dir)
    files_source = glob.glob(class_dir + '/**/*.flac', recursive=True)

    source_audio = torch.zeros(1)  # Initialize with a tensor of zeros
    while torch.all(source_audio == 0) or source_audio.shape[-1] < MIN_LENGTH_AUDIO:  # Continue until a non-zero tensor is found
        filename_source = random.choice(files_source)
        source_class = class_dir.split('/')[3]
        source_audio, _ = torchaudio.load(filename_source)
        source_audio = source_audio.reshape(-1)

    if source_class == 'Speech':
        speaker_id = int(filename_source.split('/')[6])
        speaker_gender = librispeech_metadata[speaker_id]
        if speaker_gender == 'M':
            source_class = 'male'
        else:
            source_class = 'female'

    return source_audio, source_class
