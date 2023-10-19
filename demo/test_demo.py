#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import time
import json
import yaml
import glob
import logging
import argparse
import subprocess
import typing as T

import torch
import torchaudio
from tqdm import tqdm
from scipy.signal import fftconvolve
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC

from nvas3d.data_loader.data_loader import SSAVDatasetQueryAll
from nvas3d.model.model import NVASNet
from nvas3d.utils.audio_utils import stft_to_audio, spec2wav, compute_metrics_si
from nvas3d.utils.utils import overlap_chunk


def dereverberate(
    model: torch.nn.Module,
    data: dict,
    target_length: int = 256
) -> tuple:
    """
    Applies the model to long audio (expected batch size to be 1) for dereverberation.

    Args:
    - model:                    The neural network model for dereverberation.
    - data:                     A dictionary containing input data such as 'input_stft', 'rgb', and 'depth'.
    - target_length:            The target length for STFT frames. Default is 256.

    Returns:
    - source_detection_score:   Detection results: Score of source detection.
    - pred_audio:               Separation and dereverberation results: Estimated dry sound.
    """

    _, num_receivers_double, n_freq, n_frame = data['input_stft'].shape  # [n_batch, M * 2, F, N_long]
    num_receivers = num_receivers_double // 2
    tgt_shape = (n_freq - 1, target_length)
    size, step, left_padding = tgt_shape[1], tgt_shape[1] // 2, tgt_shape[1] // 4
    input_stft_multi = data['input_stft'][0].reshape(num_receivers, 2, n_freq, n_frame)

    overlapped_spec_list = []
    for input_stft in input_stft_multi:
        overlapped_spec = overlap_chunk(input_stft[:, 1: tgt_shape[0] + 1, :], 2, size, step, left_padding)
        overlapped_spec = overlapped_spec.permute(2, 0, 1, 3)  # [B, 2, F, N] (chunk audio into multiple batches)
        overlapped_spec_list.append(overlapped_spec)
    input_stft = torch.stack(overlapped_spec_list, dim=1)  # [B, M, 2, F, N]

    B = overlapped_spec.size(0)

    visual_sensors = []
    inputs = dict()
    if 'rgb' in data.keys():
        inputs['rgb'] = data['rgb']
        visual_sensors.append('rgb')
        inputs['depth'] = data['depth']
        visual_sensors.append('depth')

    for visual_sensor in visual_sensors:
        inputs[visual_sensor] = inputs[visual_sensor].repeat(B, 1, 1, 1)

    # Perform dereverb
    input_stft = input_stft.reshape(B, num_receivers * 2, tgt_shape[0], tgt_shape[1])
    with torch.no_grad():
        inputs.update({'input_stft': input_stft, 'distance': None})
        output = model(inputs)

    # Post-process output
    pred_stft = output['pred_stft']  # [B, 2, F, N]
    pred_stft = pred_stft[:, :, :, left_padding:left_padding + step].permute(2, 0, 3, 1).reshape(tgt_shape[0], -1, 2)
    full_pred_spec = input_stft_multi[0].permute(1, 2, 0).clone()
    full_pred_spec[1: tgt_shape[0] + 1, :, :] = pred_stft[:, :n_frame, :]

    pred_audio = spec2wav(full_pred_spec)
    source_detection_score = torch.sigmoid(output['source_detection'])

    return source_detection_score.mean(dim=0), pred_audio


def find_latest_checkpoint(checkpoint_dir: str) -> T.Optional[str]:
    """
    Finds the latest checkpoint file in a given directory.

    Args:
    - checkpoint_dir: Directory where checkpoint files are stored.

    Returns:
    - The path to the latest checkpoint file or None if no checkpoint files are found.
    """

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '**', 'checkpoint_*.pt'), recursive=True)
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        return latest_checkpoint
    else:
        return None


def save_metrics_to_file(save_dir: str, metrics: dict):
    """
    Save metrics to a JSON file.

    Args:
    - save_dir: The directory where the metrics should be saved.
    - metrics:  A dictionary containing the metrics to save.
    """

    metrics_data = {
        'metrics': metrics,
    }

    save_path = os.path.join(save_dir, 'metrics.json')
    with open(save_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)

    logging.info(f'Metrics saved to {save_path}')


def safe_add(metric_name: str, current_value: float, new_value: float, valid_samples_dict: dict) -> float:
    """
    Adds the new_value to the current_value, ignoring nan or inf.

    Args:
    - metric_name:        Name of the metric being processed.
    - current_value:      Current value of the metric.
    - new_value:          New value to add to the current metric value.
    - valid_samples_dict: Dictionary that keeps track of valid samples for each metric.

    Returns:
    - Updated metric value.
    """

    if not torch.isinf(torch.tensor(new_value)) and not torch.isnan(torch.tensor(new_value)):
        valid_samples_dict[metric_name] += 1
        return current_value + new_value
    return current_value


def test_queryall(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str,
    checkpoint_path: str,
    render_map: bool = False,
    sample_rate=48000
):
    """
    Tests the NVAS model on a given dataset.

    Args:
    - model:            The model instance to test.
    - dataloader:       Dataloader for fetching test data.
    - device:           The device (cpu or cuda) on which to run the model.
    - save_dir:         Directory to save test results.
    - checkpoint_path:  Path to the model checkpoint file.
    - render_map:       Whether to render and save heatmap or not. Default is False.

    Save:
    {results_demo} = results/nvas3d_demo/default/demo/{room}/0
    │
    ├── results_drysound/
    │   ├── dry1_estimated.wav         : Estimated dry sound for source 1 location.
    │   ├── dry2_estimated.wav         : Estimated dry sound for source 2 location.
    │   ├── dry1_gt.wav                : Ground-truth dry sound for source 1.
    │   ├── dry2_gt.wav                : Ground-truth dry sound for source 2.
    │   └── detected/
    │       └── dry_{query_idx}.wav    : Estimated dry sound for query idx if positive.
    │
    ├── baseline_dsp/
    │   ├── deconv_and_sum1.wav        : Baseline result of dry sound for source 1.
    │   └── deconv_and_sum2.wav        : Baseline result of dry sound for source 2.
    │
    ├── results_detection/
    │   ├── detection_heatmap.png      : Heatmap visualizing source detection results.
    │   └── metadata.pt                : Detection results and metadata.
    │
    └── metrics.json                   : Quantitative metrics.
    """

    since = time.time()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Metrics list
    metrics_names = [
        'auc', 'pred_accuracy', 'pred_accuracy_positive', 'pred_accuracy_negative',  # detection
        'dry1_psnr', 'dry1_sdr', 'dry2_psnr', 'dry2_sdr',  # dry sound (separation and dereverberation)
        'reverb1_psnr', 'reverb1_sdr', 'reverb2_psnr', 'reverb2_sdr'  # reverberant sound (separation)
    ]

    # Initialize metrics and counters
    total_metrics = {metric: 0 for metric in metrics_names}
    valid_samples = {metric: 0 for metric in metrics_names}

    with tqdm(dataloader, unit="batch", desc="Processing", leave=True) as pbar:
        for batch_data in pbar:
            # Extract relevant data from the batch
            sample, metadata, room, room_id, *query_data = batch_data
            source1_idx, source2_idx = metadata['source1_idx'], metadata['source2_idx']
            source_idx_list = [source1_idx, source2_idx]
            receiver_idx_list = metadata['receiver_idx_list']
            grid_points = metadata['grid_points']
            grid_distance = metadata['grid_distance']

            # Logging information
            print(f'number of queries: {len(query_data[0])}')
            room = room[0]
            room_id = int(room_id[0])

            # Process and load data
            results_detection_score = []
            results_dry = []
            query_stft, query_rgb, query_depth, query_idx_list, query_positive, query_audio = query_data

            # Main loop: source detection, separation, and dereverberation for each query
            for idx, (input_stft, rgb, depth) in enumerate(zip(query_stft, query_rgb, query_depth)):
                data_dict = {
                    'input_stft': input_stft.to(device=device, dtype=torch.float)
                }

                # If visual data is available, add it to the data_dict
                if rgb.any():
                    data_dict.update({
                        'rgb': rgb.to(device=device, dtype=torch.float),
                        'depth': depth.to(device=device, dtype=torch.float)
                    })

                # Our network
                with torch.no_grad():
                    # Detection
                    pred_detection_score, _ = dereverberate(model, data_dict)

                    # Dry sound estimation (w/ adjusted audio length)
                    len_stft = data_dict['input_stft'].shape[-1]
                    new_len_stft = (len_stft // 256) * 256
                    data_dict['input_stft'] = data_dict['input_stft'][..., :new_len_stft]

                    output = model(data_dict, disable_detection=True)

                # Convert output STFT to wav
                pred_stft = output['pred_stft'][0]
                pred_dry = stft_to_audio(pred_stft, data_dict['input_stft'][0]).detach().cpu()

                # Detection and dry-sound estimation results
                results_detection_score.append(pred_detection_score)
                results_dry.append(pred_dry)

                # Clear CUDA cache periodically to free memory
                if idx % 10 == 0:
                    torch.cuda.empty_cache()

            # Process detections and dry sounds
            detection_results = []
            ground_truth_detections = []
            dry_sound_detected = {}

            # Extract dry sounds and detection results for each query index
            for query_idx, detection, dry_sound, current_query_audio in zip(query_idx_list, results_detection_score, results_dry, query_audio):
                is_gt_detected = query_idx[0] in source_idx_list
                is_pred_detected = detection[0] > 0.5

                detection_results.append(is_pred_detected == is_gt_detected)
                ground_truth_detections.append(is_gt_detected)

                # Dry sound at GT source location for reference
                if query_idx == source1_idx:
                    dry1 = dry_sound
                    wiener1 = current_query_audio
                elif query_idx == source2_idx:
                    dry2 = dry_sound
                    wiener2 = current_query_audio

                # Dry sound at detected source location
                if is_pred_detected:
                    dry_sound_detected[query_idx] = dry_sound

            # Compute detection accuracy metrics
            pred_tensor = torch.tensor(results_detection_score)
            target_tensor = torch.tensor(ground_truth_detections)

            auc_metric = BinaryAUROC(thresholds=None)
            auc_score = auc_metric(pred_tensor, target_tensor)

            detection_result_tensor = torch.tensor(detection_results).float()
            pred_accuracy_overall = detection_result_tensor.mean(dim=0)
            pred_accuracy_positive = detection_result_tensor[torch.tensor(query_positive)].mean(dim=0)
            pred_accuracy_negative = detection_result_tensor[~torch.tensor(query_positive)].mean(dim=0)

            # Compute reverberant sound
            m = 0  # first microphone
            reverb1_audio = sample['reverb1_audio'][0][0]
            ir1_m = sample['ir1'][0][m]
            pred_reverb1 = torch.from_numpy(fftconvolve(dry1, ir1_m)[:len(reverb1_audio)])

            reverb2_audio = sample['reverb2_audio'][0][0]
            ir2_m = sample['ir2'][0][m]
            pred_reverb2 = torch.from_numpy(fftconvolve(dry2, ir2_m)[:len(reverb2_audio)])

            # Evaluate predicted dry sounds against ground-truth sounds
            len_pred = pred_dry.shape[-1]
            receiver_audio = sample['receiver_audio'][0][:len_pred]
            source1_audio = sample['source1_audio'][0][:len_pred]
            source2_audio = sample['source2_audio'][0][:len_pred]

            dry1_metrics = compute_metrics_si(dry1, source1_audio)
            dry2_metrics = compute_metrics_si(dry2, source2_audio)
            reverb1_metrics = compute_metrics_si(pred_reverb1, reverb1_audio)
            reverb2_metrics = compute_metrics_si(pred_reverb2, reverb2_audio)

            # Extract metrics into a dictionary for easy access
            metrics_keys = ['psnr', 'sdr']  # ['psnr', 'sdr', 'sipsnr', 'sisdr']
            new_values = {
                'auc': auc_score.item(),
                'pred_accuracy': pred_accuracy_overall.item(),
                'pred_accuracy_positive': pred_accuracy_positive.item(),
                'pred_accuracy_negative': pred_accuracy_negative.item(),
            }
            for idx, key in enumerate(metrics_keys):
                new_values[f'dry1_{key}'] = dry1_metrics[idx].item()
                new_values[f'dry2_{key}'] = dry2_metrics[idx].item()
                new_values[f'reverb1_{key}'] = reverb1_metrics[idx].item()
                new_values[f'reverb2_{key}'] = reverb2_metrics[idx].item()

            # Save metrics
            id_dir = os.path.join(save_dir, str(room), str(room_id))
            os.makedirs(id_dir, exist_ok=True)
            save_metrics_to_file(id_dir, new_values)

            # Save audio files
            output_dir = os.path.join(id_dir, 'results_drysound')
            os.makedirs(f'{output_dir}/detected', exist_ok=True)

            # Dry sound at GT source locations (for audio quality comparison with gt)
            torchaudio.save(os.path.join(output_dir, 'dry1_estimated.wav'), dry1.unsqueeze(0), sample_rate)
            torchaudio.save(os.path.join(output_dir, 'dry2_estimated.wav'), dry2.unsqueeze(0), sample_rate)
            torchaudio.save(os.path.join(output_dir, 'dry1_gt.wav'), source1_audio.unsqueeze(0), sample_rate)
            torchaudio.save(os.path.join(output_dir, 'dry2_gt.wav'), source2_audio.unsqueeze(0), sample_rate)

            # Dry sound at detected source locations (for novel-view synthesis) (don't know which one is which source)
            detected_source_idx = []
            for query_idx, dry_sound in dry_sound_detected.items():
                torchaudio.save(os.path.join(output_dir, f'detected/dry_{query_idx.item()}.wav'), dry_sound.unsqueeze(0), sample_rate)
                detected_source_idx.append(query_idx.item())
            metadata['detected_source_idx'] = detected_source_idx

            # DSP Baseline (deconv-and-sum)
            wiener1_mean = wiener1[0].mean(dim=0)
            wiener2_mean = wiener2[0].mean(dim=0)

            baseline_dsp_dir = os.path.join(id_dir, 'baseline_dsp')
            os.makedirs(baseline_dsp_dir, exist_ok=True)

            torchaudio.save(os.path.join(baseline_dsp_dir, 'deconv_and_sum1.wav'), wiener1_mean.unsqueeze(0), sample_rate)
            torchaudio.save(os.path.join(baseline_dsp_dir, 'deconv_and_sum2.wav'), wiener2_mean.unsqueeze(0), sample_rate)

            # Prepare Detection results and heatmap
            prediction_list = []
            count = 0
            for i_point in range(grid_points.shape[1]):
                if i_point in query_idx_list:
                    prediction_list.append(results_detection_score[count].item())
                    count += 1
                else:
                    prediction_list.append(0.0)
            metadata['prediction_list'] = prediction_list

            # Save detection results and metadata for the scene
            output_dir = os.path.join(id_dir, 'results_detection')
            os.makedirs(output_dir, exist_ok=True)
            torch.save(metadata, os.path.join(output_dir, 'metadata.pt'))

            if render_map:
                def run_function(function_name, *args):  # use subprocess because of memory leak in soundspaces
                    subprocess.run(["python", "soundspaces_nvas3d/utils/render_scene_script.py", function_name, *map(str, args)])
                source_idx_list = [x.item() for x in source_idx_list]
                receiver_idx_list = [x.item() for x in receiver_idx_list]
                run_function("heatmap", f'{output_dir}/detection_heatmap.png', room, grid_points[0].tolist(), prediction_list, receiver_idx_list, grid_distance)

            # Aggregate metrics
            for metric in metrics_names:
                total_metrics[metric] = safe_add(metric, total_metrics[metric], new_values[metric], valid_samples)

            curr_avg_metrics = {metric: (total_metrics[metric] / valid_samples[metric]) if valid_samples[metric] else 0 for metric in metrics_names}

            # Update tqdm with the current metrics
            pbar.set_postfix(Acc=f"{curr_avg_metrics['pred_accuracy']:.2f}",
                             PSNR=f"{curr_avg_metrics['dry1_psnr']:.2f}",
                             SDR=f"{curr_avg_metrics['dry1_sdr']:.2f}")

            avg_metrics = {metric: (total_metrics[metric] / valid_samples[metric]) if valid_samples[metric] else 0 for metric in metrics_names}
            time_elapsed = time.time() - since

            logging.info(f'Test complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            logging.info('[Metrics]:')
            for metric, value in avg_metrics.items():
                logging.info(f"{metric}: {value:.4f}")

            # # Save average metrics
            # save_metrics_to_file(save_dir, avg_metrics)

            print('Test finished.')


def main(args):
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    exp_dir = f'nvas3d/assets/saved_models/{args.model}'

    # Load configuration
    config_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.dataset_dir:
        config['data_loader']['nvas3d_dataset'] = args.dataset_dir
        config['data_loader']['audio_format'] = args.audio_format

    # Derive directory paths
    expname = os.path.basename(exp_dir)
    nvas3d_dataset_queryall = config['data_loader']['nvas3d_dataset']
    save_dir = os.path.join('results', nvas3d_dataset_queryall, expname, args.test_mode)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize DataLoader with dataset
    data_loader_args = config['data_loader']
    dataset = SSAVDatasetQueryAll(
        args.test_mode,
        config['use_visual'],
        config['use_deconv'],
        data_loader_args['num_receivers'],
        data_loader_args['hop_length'],
        data_loader_args['win_length'],
        data_loader_args['n_fft'],
        nvas3d_dataset_queryall,
        data_loader_args['audio_format']
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize model and load checkpoint
    model = NVASNet(
        data_loader_args['num_receivers'],
        config['use_visual']
    )
    model = model.to(device)

    checkpoint_path = find_latest_checkpoint(os.path.join(exp_dir, 'checkpoints'))
    test_queryall(model, dataloader, device, save_dir, checkpoint_path, render_map=True)


if __name__ == "__main__":
    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    # Config parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='default', help='Experiment name')  # 20230901_233146_1
    parser.add_argument('--test_mode', type=str, default='demo', help='val or test or demo')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--dataset_dir', type=str, default='nvas3d_demo', help='Dataset for test')
    parser.add_argument('--audio_format', type=str, default='flac', help='Dataset for test')
    args = parser.parse_args()

    main(args)
