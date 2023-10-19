#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import typing as T

import torch
import torch.nn.functional as F
from torchmetrics.audio import SignalDistortionRatio
from torchmetrics import ScaleInvariantSignalDistortionRatio


EPS = 1e-7


def cosine_similarity(
    audio1: torch.Tensor,
    audio2: torch.Tensor
) -> float:
    """
    Calculate the cosine similarity between two audio signals.

    Args:
    - audio1: The first audio signal.
    - audio2: The second audio signal.

    Returns:
    - The cosine similarity between audio1 and audio2.
    """

    cs_value = F.cosine_similarity(audio1.reshape(1, -1), audio2.reshape(1, -1), dim=1)
    return cs_value.item()


def cs_list(audio_list: T.List[torch.Tensor]) -> float:
    """
    Calculate the average cosine similarity between all pairs of audio signals in a list.

    Args:
    - audio_list: List of audio signals.

    Returns:
    - Average cosine similarity between all pairs of audio signals in the list.
    """

    total_cs = 0
    n_pair = 0

    for i in range(len(audio_list)):
        for j in range(i + 1, len(audio_list)):
            cs_value = cosine_similarity(audio_list[i], audio_list[j])
            total_cs += cs_value
            n_pair += 1

    return total_cs / n_pair


def cs_tensor(audio_tensor: torch.Tensor) -> float:
    """
    Calculate the average cosine similarity between all pairs of audio signals in a tensor.

    Args:
    - audio_tensor: Tensor of audio signals.

    Returns:
    - Average cosine similarity between all pairs of audio signals in the tensor.
    """

    audio_tensor = audio_tensor[0]  # first batch
    audio_list = [audio_tensor[i].unsqueeze(0) for i in range(audio_tensor.size(0))]
    total_cs = 0
    n_pair = 0

    for i in range(len(audio_list)):
        for j in range(i + 1, len(audio_list)):
            cs_value = cosine_similarity(audio_list[i], audio_list[j])
            total_cs += cs_value
            n_pair += 1

    return total_cs / n_pair


def compute_metrics(pred_audio: torch.Tensor, ref_audio: torch.Tensor) -> T.Tuple[float, float]:
    """
    Compute PSNR and SDR metrics between predicted and reference audio signals.

    Args:
    - pred_audio:   Predicted audio signal.
    - ref_audio:    Reference audio signal.

    Returns:
    - PSNR and SDR scores between predicted and reference audio.
    """

    pred_audio = pred_audio.reshape(-1).detach().cpu()
    ref_audio = ref_audio.reshape(-1).detach().cpu()
    assert len(pred_audio) == len(ref_audio)

    compute_sdr = SignalDistortionRatio()
    psnr_score = psnr(pred_audio, ref_audio)
    try:
        sdr_score = compute_sdr(pred_audio, ref_audio)
    except:
        sdr_score = torch.tensor(0.0)

    return psnr_score, sdr_score


def compute_metrics_si(
    pred_audio: torch.Tensor,
    ref_audio: torch.Tensor,
    compute_all: bool = False
) -> T.Tuple[float, float, float, float]:
    """
    Compute PSNR, SDR, SI-PSNR, and SI-SDR metrics between predicted and reference audio signals.

    Args:
    - pred_audio:   Predicted audio signal.
    - ref_audio:    Reference audio signal.
    - compute_all:  If True, compute all metrics. Otherwise, compute only PSNR and SDR.

    Returns:
    - PSNR, SDR, SI-PSNR, and SI-SDR scores between predicted and reference audio.
    """

    pred_audio = pred_audio.reshape(-1).detach().cpu()
    ref_audio = ref_audio.reshape(-1).detach().cpu()
    assert len(pred_audio) == len(ref_audio)

    compute_sdr = SignalDistortionRatio()
    psnr_score = psnr(pred_audio, ref_audio)
    try:
        sdr_score = compute_sdr(pred_audio, ref_audio)
    except:
        sdr_score = torch.tensor(0.0)

    if compute_all:
        compute_sisdr = ScaleInvariantSignalDistortionRatio()
        sipsnr_score = psnr(pred_audio / pred_audio.abs().max(), ref_audio / ref_audio.abs().max())

        try:
            sisdr_score = compute_sisdr(pred_audio, ref_audio)
        except:
            sisdr_score = torch.tensor(0.0)

    else:
        sipsnr_score = torch.tensor(0.0)
        sisdr_score = torch.tensor(0.0)

    return psnr_score, sdr_score, sipsnr_score, sisdr_score


def wav2spec(
    wav: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 480,
    win_length: int = 1200
) -> torch.Tensor:
    """
    Transforms waveforms to STFT using a Hamming window.

    Args:
    - wav:          Input waveform of shape [B, T].
    - n_fft:        FFT window size.
    - hop_length:   Hop length for STFT
    - win_length:   Window length for STFT

    Returns:
    - STFT of shape [B, F, N, 2].
    """

    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                      window=torch.hamming_window(win_length, device=wav.device), pad_mode='constant',
                      return_complex=False)

    return stft


def spec2wav(
    spec: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 480,
    win_length: int = 1200
) -> torch.Tensor:
    """
    Transforms STFT back to waveforms using inverse STFT with a Hamming window.

    Args:
    - spec:         Input STFT spectrogram of shape [B, F, N, 2].
    - n_fft:        FFT size.
    - hop_length:   Hop length for STFT
    - win_length:   Window length for STFT

    Returns:
    - Waveform of shape [B, T].
    """

    wav = torch.istft(spec, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=torch.hamming_window(win_length).to(spec.device, dtype=torch.float))
    return wav


def stft_to_audio(
    pred_stft: torch.Tensor,
    input_stft: torch.Tensor
) -> torch.Tensor:
    """
    Convert predicted STFT to audio, preserving the DC component of the input STFT.

    Args:
    - pred_stft:    Predicted STFT.
    - input_stft:   Input STFT used for DC component.

    Returns:
    - Converted audio tensor.
    """

    # use dc component of input stft
    full_pred_stft = input_stft.clone().permute(1, 2, 0)[..., :2]

    # pred
    pred_stft = pred_stft.permute(1, 2, 0)
    full_pred_stft[1:, :, :] = pred_stft  # copy dc from input
    pred_audio = spec2wav(full_pred_stft)

    return pred_audio


def wiener_deconv_batch(signal: torch.Tensor,
                        kernel: torch.Tensor,
                        snr: float,
                        is_cpu: bool = False
                        ) -> torch.Tensor:
    """
    Applies Wiener deconvolution on a batch of signals.

    Args:
        signal:     Batch of input signals.
        kernel:     Deconvolution kernel.
        snr:        Signal-to-noise ratio.
        is_cpu:     Flag to run calculations on CPU (default: False, runs on GPU).

    Returns: 
    - Batch of filtered signals.
    """

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

    return filtered_signal


def psnr(target, reference):
    """
    Calculates the PSNR between a reference and a target signal.

    Args:
        reference:  Reference signal.
        target:     Target signal.

    Returns: 
    - PSNR value.
    """

    reference, target = clip_two(reference, target)

    mse = torch.mean((reference - target)**2)
    max_squared_value = torch.max(reference)**2
    return 10 * torch.log10(max_squared_value / mse)


def clip_two(audio1, audio2):
    """
    Clips two audio signals to the same length.

    Args:
        audio1: First audio signal.
        audio2: Second audio signal.

    Returns: 
    - Two audio signals of the same length.
    """

    length_diff = audio1.shape[-1] - audio2.shape[-1]

    if length_diff == 0:
        return audio1, audio2
    elif length_diff > 0:
        audio1 = audio1[..., :audio2.shape[-1]]
    elif length_diff < 0:
        audio2 = audio2[..., :audio1.shape[-1]]

    return audio1, audio2


def clip_all(audio_list):
    """
    Clips all audio signals in a list to the same length.

    Args: 
        audio_list: List of audio signals.

    Returns: 
    - List of audio signals of the same length.
    """

    min_length = min(audio.shape[-1] for audio in audio_list)
    clipped_audio_list = []
    for audio in audio_list:
        clipped_audio = audio[..., :min_length]
        clipped_audio_list.append(clipped_audio)

    return clipped_audio_list


def pad_all(audio_list):
    """
    Pads all audio signals in a list to the same length.

    Args: 
        audio_list: List of audio signals.

    Returns: 
    - List of audio signals of the same length.
    """

    max_length = max(audio.shape[-1] for audio in audio_list)
    padded_audio_list = []
    for audio in audio_list:
        padding = max_length - audio.shape[-1]
        padded_audio = F.pad(audio, (0, padding))
        padded_audio_list.append(padded_audio)

    return padded_audio_list


def pad_two(audio1, audio2):
    """
    Pads two audio signals to the same length.

    Args: 
        audio1: First audio signal.
        audio2: Second audio signal.

    Returns: 
    - Two audio signals of the same length.
    """

    length_diff = audio1.shape[-1] - audio2.shape[-1]

    if length_diff == 0:
        return audio1, audio2
    elif length_diff > 0:
        audio2 = F.pad(audio2, (0, length_diff), value=0)
    elif length_diff < 0:
        audio1 = F.pad(audio1, (0, -length_diff), value=0)

    return audio1, audio2


def create_segments(input_tensor, T_seg, overlap_ratio):
    # input_tensor is of shape [M, T]
    # T_seg is the desired size of each segment
    # overlap_ratio is the desired ratio of overlap between segments

    # Check that input is a 2D tensor
    assert len(input_tensor.shape) == 2, "Input tensor must be 2D"
    M, T = input_tensor.shape

    # If T_seg is not an integer, or greater than T, the tensor cannot be segmented as specified
    assert T_seg > 0 and T_seg <= T, "Invalid segment size"

    # Calculate the step size between segments
    step = round(T_seg * (1 - overlap_ratio))
    assert step > 0, "Step size must be greater than 0"

    # Create the overlapping segments
    segments = input_tensor.unfold(1, T_seg, step)

    # Move the segments dimension to the first dimension
    segments = segments.permute(1, 0, 2)

    return segments


def create_segments_batch(input_tensor, T_seg, overlap_ratio):
    # input_tensor is of shape [B, M, T]
    # T_seg is the desired size of each segment
    # overlap_ratio is the desired ratio of overlap between segments

    # Check that input is a 3D tensor
    assert len(input_tensor.shape) == 3, "Input tensor must be 3D"
    B, M, T = input_tensor.shape

    # If T_seg is not an integer, or greater than T, the tensor cannot be segmented as specified
    assert T_seg > 0 and T_seg <= T, "Invalid segment size"

    # Calculate the step size between segments
    step = round(T_seg * (1 - overlap_ratio))
    assert step > 0, "Step size must be greater than 0"

    # Create the overlapping segments
    segments = input_tensor.unfold(2, T_seg, step)

    # Move the segments dimension to the second dimension
    segments = segments.permute(0, 2, 1, 3)

    return segments  # [B, N, M, T_seg]


def deconv_and_sum(y: torch.Tensor,  # [M, Ty] #
                   h: torch.Tensor,  # [M, Th] #,
                   snr: float = 100.0,
                   is_debug: bool = False
                   ) -> torch.Tensor:
    """
    deconv and average in frequency domain 
    """

    M = y.shape[0]
    n_fft = y.shape[-1] + h.shape[-1] - 1

    # sterring vector
    v = torch.fft.rfft(h, n=n_fft).permute(1, 0).unsqueeze(-1)  # [F, M, 1]

    # deconv-and-mean
    snr_normalized = snr / abs(h).max()
    W = torch.conj(v) / (torch.abs(v)**2 + 1 / snr_normalized) / M
    # W_mean = torch.conj(v) / (torch.abs(v)**2) / M
    W = W.transpose(-2, -1)
    w = torch.fft.irfft(W.reshape(-1, M), n=h.shape[-1], dim=0)  # [T, M]

    # convolution
    Y_fft = torch.fft.rfft(y, n=n_fft)  # [M, F]
    DM_fft = (Y_fft * W.reshape(-1, M).permute(1, 0))
    deconvmean_m_list = []
    for DM_fft_m in DM_fft:
        dm_m = torch.fft.irfft(DM_fft_m, n=n_fft)  # [T, M]
        deconvmean_m_list.append(dm_m)
    deconvmean = torch.fft.irfft(DM_fft.sum(dim=0), n=n_fft)  # [T]

    if is_debug:
        M = y.shape[0]
        n_fft = y.shape[-1] + h.shape[-1] - 1
        overlap_ratio = 0.99
        rho = 0.01

        y_segments = create_segments(y, h.shape[-1], overlap_ratio)  # [N, M, Th]
        Y = torch.fft.rfft(y_segments, n=n_fft).permute(2, 1, 0)  # [F, M, N]

        # spectral matrix
        R = torch.matmul(Y, Y.transpose(-2, -1).conj())  # [F, M, M]
        rho_diag = torch.diag_embed(torch.full((R.shape[0], R.shape[1]), rho, device=R.device, dtype=R.dtype))
        R = R + rho_diag
        # spectral matrix
        R = torch.matmul(Y, Y.transpose(-2, -1).conj())  # [F, M, M]
        rho_diag = torch.diag_embed(torch.full((R.shape[0], R.shape[1]), rho, device=R.device, dtype=R.dtype))
        R = R + rho_diag

        print('[deconv_and_sum]')
        print(f'power = {torch.matmul(torch.matmul(W, R), W.transpose(-2, -1).conj()).abs().mean()}')
        print(f'constraint: real = {torch.matmul(W, v).real.mean()}, imag = {torch.matmul(W, v).imag.mean()}')

    return deconvmean, deconvmean_m_list, w.permute(1, 0)  # [M, T]


def fft_conv(signal: torch.Tensor,
             kernel: torch.Tensor,
             is_cpu: bool = False
             ) -> torch.Tensor:
    """
    Convolution using FFT
    """

    if is_cpu:
        signal = signal.detach().cpu()
        kernel = kernel.detach().cpu()

    padded_signal = F.pad(signal.reshape(-1), (0, kernel.size(-1) - 1))
    padded_kernel = F.pad(kernel.reshape(-1), (0, signal.size(-1) - 1))

    signal_fr = torch.fft.rfftn(padded_signal, dim=-1)
    kernel_fr = torch.fft.rfftn(padded_kernel, dim=-1)

    output_fr = signal_fr * kernel_fr
    output = torch.fft.irfftn(output_fr, s=padded_signal.shape[-1], dim=-1)

    return output
