#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import numpy as np
import typing as T

from scipy.signal import oaconvolve


def setup_dynamic_interp(
    receiver_position: np.ndarray,
    total_samples: int,
) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Setup moving path with a constant speed for a receiver, given its positions in 3D space.

    Args:
    - receiver_position:    Receiver positions in 3D space of shape (num_positions, 3).
    - total_samples:        Total number of samples in the audio.

    Returns:
    - interp_index:         Indices representing the start positions for interpolation.
    - interp_weight:        Weight values for linear interpolation.
    """

    # Calculate the number of samples per interval
    distance = np.linalg.norm(np.diff(receiver_position, axis=0), axis=1)
    speed_per_sample = distance.sum() / total_samples
    samples_per_interval = np.round(distance / speed_per_sample).astype(int)

    # Distribute rounding errors
    error = total_samples - samples_per_interval.sum()
    for i in np.random.choice(len(samples_per_interval), abs(error)):
        samples_per_interval[i] += np.sign(error)

    # Calculate indices and weights for linear interpolation
    interp_index = np.repeat(np.arange(len(distance)), samples_per_interval)
    interp_weight = np.concatenate([np.linspace(0, 1, num, endpoint=False) for num in samples_per_interval])

    return interp_index, interp_weight.astype(np.float32)


def convolve_moving_receiver(
    source_audio: np.ndarray,
    rirs: np.ndarray,
    interp_index: T.List[int],
    interp_weight: T.List[float]
) -> np.ndarray:
    """
    Apply convolution between an audio signal and moving impulse responses (IRs).

    Args:
    - source_audio:     Source audio of shape (audio_len,)
    - rirs:             RIRs of shape (num_positions, num_channels, ir_length)
    - interp_index:     Indices representing the start positions for interpolation of shape (audio_len,).
    - interp_weight:    Weight values for linear interpolation of shape (audio_len,).

    Returns:
    - Convolved audio signal of shape (num_channels, audio_len)
    """

    num_channels = rirs.shape[1]
    audio_len = source_audio.shape[0]

    # Perform convolution for each position and channel
    convolved_audios = oaconvolve(source_audio[None, None, :], rirs, axes=-1)[..., :audio_len]

    # NumPy fancy indexing and broadcasting for interpolation
    start_audio = convolved_audios[interp_index, np.arange(num_channels)[:, None], np.arange(audio_len)]
    end_audio = convolved_audios[interp_index + 1, np.arange(num_channels)[:, None], np.arange(audio_len)]
    interp_weight = interp_weight[None, :]

    # Apply linear interpolation
    moving_audio = (1 - interp_weight) * start_audio + interp_weight * end_audio

    return moving_audio
