#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchaudio
from torchmetrics.audio import SignalDistortionRatio

from nvas3d.utils.audio_utils import psnr, clip_two


def save_specgram(filename, stft):
    mag = torch.sqrt(stft[..., 0]**2 + stft[..., 1]**2)
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    cax1 = ax.imshow(torch.log10(mag), aspect='auto', origin='lower')
    fig.colorbar(cax1, ax=ax, orientation='horizontal')
    ax.set_title('Magnitude spectrogram')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def plot_specgram(filename, waveform, sample_rate, plot_phase=False):

    mag = compute_spectrogram(waveform, use_mag=True).squeeze()
    phase = compute_spectrogram(waveform, use_phase=True).squeeze()

    phase = phase * (180 / np.pi)

    if plot_phase:
        fig, ax = plt.subplots(2, 1, figsize=(14, 8))
        cax1 = ax[0].imshow(torch.log10(mag), aspect='auto', origin='lower')
        fig.colorbar(cax1, ax=ax[0], orientation='horizontal')
        ax[0].set_title('Magnitude spectrogram')

        cax2 = ax[1].imshow(phase, aspect='auto', origin='lower', cmap='twilight')
        fig.colorbar(cax2, ax=ax[1], orientation='horizontal')
        ax[1].set_title('Phase spectrogram')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        cax1 = ax.imshow(torch.log10(mag), aspect='auto', origin='lower')
        fig.colorbar(cax1, ax=ax, orientation='horizontal')
        ax.set_title('Magnitude spectrogram')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def plot_waveform(filename, waveform, sample_rate):
    plt.figure(figsize=(14, 4))
    plt.plot(np.linspace(0, len(waveform[0]) / sample_rate, len(waveform[0])), waveform[0])
    plt.title('Waveform')
    plt.xlabel('Time [s]')
    plt.savefig(filename)
    plt.close()


def plot_debug(filename, waveform, sample_rate, save_png=False, save_normalized=False, reference=None):
    waveform = waveform.reshape(1, -1)

    if reference is not None:
        reference = reference.reshape(1, -1)
        waveform, reference = clip_two(waveform, reference)
        # assert waveform.shape[-1] == reference.shape[-1]
        psnr_score = psnr(reference.detach().cpu(), waveform.detach().cpu())
        compute_sdr = SignalDistortionRatio()
        sdr = compute_sdr(reference.detach().cpu(), waveform.detach().cpu())

    if save_normalized:
        waveform /= waveform.abs().max()

    if save_png:
        plot_specgram(f'{filename}_specgram.png', waveform, sample_rate)
        plot_waveform(f'{filename}_waveform.png', waveform, sample_rate)

    if reference is not None:
        filename = f'{filename}({psnr_score:.1f})({sdr:.1f})'
    torchaudio.save(f'{filename}.wav', waveform, sample_rate)


def compute_spectrogram(audio_data, n_fft=2048, hop_length=480, win_length=1200, use_mag=False, use_phase=False, use_complex=False):

    audio_data = to_tensor(audio_data)
    stft = torch.stft(audio_data, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                      window=torch.hamming_window(win_length, device=audio_data.device), pad_mode='constant',
                      return_complex=not use_complex)

    if use_mag:
        spectrogram = stft.abs().unsqueeze(-1)
    elif use_phase:
        spectrogram = stft.angle().unsqueeze(-1)
    elif use_complex:
        # one channel for real and one channel for imaginary
        spectrogram = stft
    else:
        raise ValueError

    return spectrogram


def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)
