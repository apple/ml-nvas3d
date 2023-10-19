#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import torch
import torchaudio
import matplotlib.pyplot as plt
import torch.nn.functional as F
import typing as T


def fft_conv(
    signal: torch.Tensor,
    kernel: torch.Tensor,
    is_cpu: bool = False
) -> torch.Tensor:
    """
    Perform convolution of a signal and a kernel using Fast Fourier Transform (FFT).

    Args:
    - signal (torch.Tensor): Input signal tensor.
    - kernel (torch.Tensor): Kernel tensor.
    - is_cpu (bool, optional): Flag to determine if the operation should be on the CPU.

    Returns:
    - torch.Tensor: Convolved signal.
    """

    if is_cpu:
        signal = signal.detach().cpu()
        kernel = kernel.detach().cpu()

    padded_signal = F.pad(signal.reshape(-1), (0, kernel.size(-1) - 1))
    padded_kernel = F.pad(kernel.reshape(-1), (0, signal.size(-1) - 1))

    signal_fr = torch.fft.rfftn(padded_signal, dim=-1)
    kernel_fr = torch.fft.rfftn(padded_kernel, dim=-1)

    output_fr = signal_fr * kernel_fr
    output = torch.fft.irfftn(output_fr, dim=-1)

    return output


def wiener_deconv(
    signal: torch.Tensor,
    kernel: torch.Tensor,
    snr: float,
    is_cpu: bool = False
) -> torch.Tensor:
    """
    Perform Wiener deconvolution on a given signal using a specified kernel.

    Args:
    - signal (torch.Tensor): Input signal tensor.
    - kernel (torch.Tensor): Kernel tensor.
    - snr (float): Signal-to-noise ratio.
    - is_cpu (bool, optional): Flag to determine if the operation should be on the CPU.

    Returns:
    - torch.Tensor: Deconvolved signal.
    """

    if is_cpu:
        signal = signal.detach().cpu()
        kernel = kernel.detach().cpu()

    n_fft = signal.shape[-1] + kernel.shape[-1] - 1
    signal_fr = torch.fft.rfft(signal.reshape(-1), n=n_fft)
    kernel_fr = torch.fft.rfft(kernel.reshape(-1), n=n_fft)

    wiener_filter_fr = torch.conj(kernel_fr) / (torch.abs(kernel_fr)**2 + 1 / snr)

    filtered_signal_fr = wiener_filter_fr * signal_fr

    filtered_signal = torch.fft.irfft(filtered_signal_fr)

    # Crop the filtered signal to the original size
    filtered_signal = filtered_signal[:signal.shape[-1]]

    return filtered_signal


def wiener_deconv_list(
    signal: T.List[torch.Tensor],
    kernel: T.List[torch.Tensor],
    snr: float,
    is_cpu: bool = False
) -> torch.Tensor:
    """
    wiener_deconv for list input.

    Args:
    - signal (torch.Tensor): List of signals.
    - kernel (torch.Tensor): List of kernels.
    - snr (float): Signal-to-noise ratio.
    - is_cpu (bool, optional): Flag to determine if the operation should be on the CPU.

    Returns:
    - torch.Tensor: Deconvolved signal.
    """

    M = len(signal)
    if isinstance(signal, list):
        signal = torch.stack(signal).reshape(M, -1)
    assert signal.shape[0] == M
    kernel = torch.stack(kernel).reshape(M, -1)
    snr /= abs(kernel).max()

    if is_cpu:
        signal = signal.detach().cpu()
        kernel = kernel.detach().cpu()

    n_batch, n_samples = signal.shape

    # Pad the signals and kernels to avoid circular convolution
    padded_signal = F.pad(signal, (0, kernel.shape[-1] - 1))
    padded_kernel = F.pad(kernel, (0, signal.shape[-1] - 1))

    # Compute the Fourier transforms
    signal_fr = torch.fft.rfft(padded_signal, dim=-1)
    kernel_fr = torch.fft.rfft(padded_kernel, dim=-1)

    # Compute the Wiener filter in the frequency domain
    wiener_filter_fr = torch.conj(kernel_fr) / (torch.abs(kernel_fr)**2 + 1 / snr)

    # Apply the Wiener filter
    filtered_signal_fr = wiener_filter_fr * signal_fr

    # Compute the inverse Fourier transform
    filtered_signal = torch.fft.irfft(filtered_signal_fr, dim=-1)

    # Crop the filtered signals to the original size
    filtered_signal = filtered_signal[:, :n_samples]

    filtered_signal_list = [filtered_signal[i] for i in range(filtered_signal.size(0))]

    return filtered_signal_list


def save_audio(
    filename: str,
    waveform: torch.Tensor,  # (ch, time) in cpu
    sample_rate: float
):
    """
    Save an audio waveform to a file.

    Args:
    - filename (str): Output filename.
    - waveform (torch.Tensor): Audio waveform tensor.
    - sample_rate (float): Sample rate of the audio.
    """

    torchaudio.save(filename, waveform, sample_rate=sample_rate)


def plot_waveform(filename: str,
                  waveform: torch.Tensor,  # (ch, time) in cpu
                  sample_rate: float = 48000,
                  title: str = None,
                  sharex: bool = True,
                  xlim: T.Tuple[float, float] = None,
                  ylim: T.Tuple[float, float] = None,
                  color: str = 'orange',
                  waveform_ref: torch.Tensor = None
                  ):
    """
    Plot an audio waveform.

    Args:
    - filename (str): Output filename.
    - waveform (torch.Tensor): Audio waveform tensor.
    - sample_rate (float, optional): Sample rate of the audio. Defaults to 48000.
    - title (str, optional): Title for the plot.
    - sharex (bool, optional): Whether to share the x-axis across subplots. Defaults to True.
    - xlim (T.Tuple[float, float], optional): Limits for x-axis.
    - ylim (T.Tuple[float, float], optional): Limits for y-axis.
    - color (str, optional): Color for the waveform plot. Defaults to 'orange'.
    - waveform_ref (torch.Tensor, optional): Reference waveform for comparison.
    """

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    if waveform_ref is not None:
        num_frames_ref = waveform_ref.shape[0]  # should be 1D
        time_axis_ref = torch.arange(0, num_frames_ref) / sample_rate

    if ylim is None:
        margin = 1.1
        ylim = (margin * waveform.min(), margin * waveform.max())

    figure, axes = plt.subplots(num_channels, 1, sharex=sharex)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        if waveform_ref is None:
            axes[c].plot(time_axis, waveform[c], color=color, linewidth=1)
        else:
            axes[c].plot(time_axis, waveform[c], color=color, alpha=0.5, linewidth=1, label='signal')
            axes[c].plot(time_axis_ref, waveform_ref, color='r', alpha=0.5, linewidth=1, label='reference')
            axes[c].legend()
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    if title is not None:
        figure.suptitle(title)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show(block=False)


def print_stats(
    waveform: torch.Tensor,
    sample_rate: T.Optional[float] = None,
    src: T.Optional[str] = None
):
    """
    Print the statistics of a given waveform.

    Args:
    - waveform (torch.Tensor): Input audio waveform tensor.
    - sample_rate (float, optional): Sample rate of the audio.
    - src (str, optional): Source of the audio, for display purposes.
    """

    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()


def plot_specgram(
    filename: str,
    waveform: torch.Tensor,
    sample_rate: float,
    title: T.Optional[str] = None,
    xlim: T.Optional[T.Tuple[float, float]] = None
):
    """
    Plot the spectrogram of a given audio waveform.

    Args:
    - filename (str): Output filename.
    - waveform (torch.Tensor): Audio waveform tensor.
    - sample_rate (float): Sample rate of the audio.
    - title (str, optional): Title for the plot.
    - xlim (T.Tuple[float, float], optional): Limits for x-axis.
    """

    waveform = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    if title is not None:
        figure.suptitle(title)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show(block=False)


def plot_debug(
    filename: str,
    waveform: torch.Tensor,
    sample_rate: float,
    save_png: bool = False
):
    """
    Generate and save waveform and spectrogram plots, and save audio waveform to a file.

    Args:
    - filename (str): Base filename for outputs.
    - waveform (torch.Tensor): Audio waveform tensor.
    - sample_rate (float): Sample rate of the audio.
    - save_png (bool, optional): Whether to save plots as PNG files. Defaults to False.
    """

    waveform = waveform.reshape(1, -1)
    if save_png:
        plot_specgram(f'{filename}_specgram.png', waveform, sample_rate)
        plot_waveform(f'{filename}_waveform.png', waveform, sample_rate)
    torchaudio.save(f'{filename}.wav', waveform, sample_rate)
