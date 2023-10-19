#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import json
import random
import argparse

import torch
import torchaudio
from tqdm import tqdm
from scipy.signal import fftconvolve
from torchmetrics.classification import BinaryAUROC

from nvas3d.utils.audio_utils import cs_list, compute_metrics_si


def normalize(audio):
    """
    Normalize the signal making the maximum of absolute value to be 1
    """
    return audio / audio.abs().max()


def compute_delayed(audio_list, delay_list):
    """
    Remove delay of audios
    """

    delayed_audio_list = []
    for audio, delay in zip(audio_list, delay_list):
        delayed_audio = torch.zeros_like(audio)
        delayed_audio[:, :-delay] = audio[:, delay:]

        delayed_audio_list.append(delayed_audio)
    return delayed_audio_list


class BaselineDSP:
    def __init__(self, split, num_receivers, audio_format, dataset_name):
        self.num_receivers = num_receivers
        self.split = split
        self.audio_format = audio_format
        self.dataset_name = dataset_name
        self.dataset_dir = f'data/{self.dataset_name}/{split}'
        self.room_id_pairs = self._get_room_id_pairs()
        # random.shuffle(self.room_id_pairs)

    def _get_room_id_pairs(self):
        pairs = []
        for room in os.listdir(self.dataset_dir):
            room_path = os.path.join(self.dataset_dir, room)
            if os.path.isdir(room_path):  # Ensure room_path is a directory
                for id in os.listdir(room_path):
                    id_path = os.path.join(room_path, id)
                    if os.path.isdir(id_path) and id != 'image':  # Ensure id_path is a directory and its name isn't 'image'
                        pairs.append((room, id))
        return pairs

    def load_audio_list(self, audio_list_name):
        audio_list = []
        for m in range(self.num_receivers):
            audio, _ = torchaudio.load(f'{audio_list_name}_{m+1}.{self.audio_format}')
            audio_list.append(audio)
        return audio_list

    def separation(self, num_eval=None):
        """
        Baseline DSP for separation using delay-and-sum and deconv-and-sum
        """

        # metrics_names = ['psnr', 'sdr', 'sipsnr', 'sisdr']
        metrics_names = ['psnr', 'sdr']
        methods = ['dry1_deconv', 'dry2_deconv', 'reverb1_deconv', 'reverb2_deconv',
                   'dry1_delay', 'dry2_delay', 'reverb1_delay', 'reverb2_delay']
        metrics = {method: {metric: [] for metric in metrics_names} for method in methods}

        loop_length = num_eval if num_eval is not None else len(self.room_id_pairs)
        for room_id_pair in tqdm(self.room_id_pairs[:loop_length]):
            room, id = room_id_pair
            data_dir = f'{self.dataset_dir}/{room}/{id}'
            # Set query: positive or negative

            metadata = torch.load(f'{data_dir}/metadata.pt')
            grid_points = metadata['grid_points']
            num_points = grid_points.shape[0]
            source1_idx = metadata['source1_idx']
            source2_idx = metadata['source2_idx']
            receiver_idx_list = metadata['receiver_idx_list']

            source1_audio, _ = torchaudio.load(f'{data_dir}/source/source1.{self.audio_format}')
            source1_audio = source1_audio[0]
            source2_audio, _ = torchaudio.load(f'{data_dir}/source/source2.{self.audio_format}')
            source2_audio = source2_audio[0]
            reverb1_audio = self.load_audio_list(f'{data_dir}/reverb/reverb1')
            reverb1_audio = reverb1_audio[0][0]
            reverb2_audio = self.load_audio_list(f'{data_dir}/reverb/reverb2')
            reverb2_audio = reverb2_audio[0][0]
            receiver_audio = self.load_audio_list(f'{data_dir}/receiver/receiver')

            ir1 = self.load_audio_list(f'{data_dir}/ir_receiver/ir1')
            ir2 = self.load_audio_list(f'{data_dir}/ir_receiver/ir2')

            for query_idx in range(num_points):
                if (query_idx not in receiver_idx_list):
                    deconv_audio = self.load_audio_list(f'{data_dir}/wiener/wiener{query_idx}')

                    deconv_and_sum = torch.stack(deconv_audio, dim=0).reshape(self.num_receivers, -1).mean(dim=0)

                    # compute delay from distance
                    c = 343
                    sample_rate = 48000
                    query_point = grid_points[query_idx]
                    delay_list = []
                    for receiver_idx in receiver_idx_list:
                        receiver_point = grid_points[receiver_idx]
                        dist = (query_point - receiver_point).norm() - 0.1  # soundspaces offset
                        delay_list.append(int(dist / c * sample_rate))

                    delayed_audio = compute_delayed(receiver_audio, delay_list)
                    delay_and_sum_0 = torch.stack(delayed_audio, dim=0).reshape(self.num_receivers, -1).mean(dim=0)

                    # # apply delay again to align with reverb1 for fair evaluation
                    # delay = delay_list[0]
                    # delay_and_sum = torch.zeros_like(delay_and_sum_0)
                    # delay_and_sum[delay:] = delay_and_sum_0[:-delay]

                    if query_idx == source1_idx:
                        dry1 = deconv_and_sum
                        dry1_delay = delay_and_sum_0

                        receiver_point = grid_points[receiver_idx_list[0]]
                        dist = (query_point - receiver_point).norm() - 0.1  # soundspaces offset
                        delay = int(dist / c * sample_rate)
                        reverb1_delay = torch.zeros_like(delay_and_sum_0)
                        reverb1_delay[delay:] = delay_and_sum_0[:-delay]

                        # reverb1
                    elif query_idx == source2_idx:
                        dry2 = deconv_and_sum
                        dry2_delay = delay_and_sum_0

                        receiver_point = grid_points[receiver_idx_list[0]]
                        dist = (query_point - receiver_point).norm() - 0.1  # soundspaces offset
                        delay = int(dist / c * sample_rate)
                        reverb2_delay = torch.zeros_like(delay_and_sum_0)
                        reverb2_delay[delay:] = delay_and_sum_0[:-delay]

            m = 0
            ir1_m = ir1[m][0]
            pred_reverb1 = torch.from_numpy(fftconvolve(dry1, ir1_m)[:len(reverb1_audio)])

            ir2_m = ir2[m][0]
            pred_reverb2 = torch.from_numpy(fftconvolve(dry2, ir2_m)[:len(reverb2_audio)])

            def compute_and_save_metrics(processed_audio, reference_audio, metric_lists, filename, save_audio=True):
                psnr_score, sdr_score, _, _ = compute_metrics_si(processed_audio, reference_audio)

                metric_lists['psnr'].append(psnr_score)
                metric_lists['sdr'].append(sdr_score)

                # Saving the processed audio
                if save_audio:
                    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensuring the directory exists
                    torchaudio.save(filename, normalize(processed_audio.unsqueeze(0)), sample_rate=48000)

            current_dir = f'test_results/{self.dataset_name}/baseline/{self.split}/{room}/{id}'
            os.makedirs(current_dir, exist_ok=True)

            # dry (deconv-and-sum)
            filename = f'{current_dir}/dry1.wav'
            compute_and_save_metrics(dry1, source1_audio, metrics['dry1_deconv'], filename)
            filename = f'{current_dir}/dry2.wav'
            compute_and_save_metrics(dry2, source2_audio, metrics['dry2_deconv'], filename)

            # dry (delay-and-sum)
            filename = f'{current_dir}/dry1_delay.wav'
            compute_and_save_metrics(dry1_delay, source1_audio, metrics['dry1_delay'], filename)
            filename = f'{current_dir}/dry2_delay.wav'
            compute_and_save_metrics(dry2_delay, source2_audio, metrics['dry2_delay'], filename)

            # reverb (deconv-and-sum)
            filename = f'{current_dir}/reverb1.wav'
            compute_and_save_metrics(pred_reverb1, reverb1_audio, metrics['reverb1_deconv'], filename)
            filename = f'{current_dir}/reverb2.wav'
            compute_and_save_metrics(pred_reverb2, reverb2_audio, metrics['reverb2_deconv'], filename)

            # reverb (delay-and-sum)
            filename = f'{current_dir}/reverb1_delay.wav'
            compute_and_save_metrics(reverb1_delay, reverb1_audio, metrics['reverb1_delay'], filename)
            filename = f'{current_dir}/reverb2_delay.wav'
            compute_and_save_metrics(reverb2_delay, reverb2_audio, metrics['reverb2_delay'], filename)

        # Print metrics
        for method, method_metrics in metrics.items():
            print(f'\n[{method} audio]')
            for metric_name, metric_values in method_metrics.items():
                tensor_values = torch.tensor(metric_values)
                is_valid = torch.isfinite(tensor_values) & ~torch.isnan(tensor_values)
                valid_values = tensor_values[is_valid]  # Exclude both 'inf' and 'nan' values
                print(f'{method} {metric_name}: {torch.tensor(valid_values).mean()}')

        # Build the dictionary with averages
        averaged_metrics = {}
        for method, method_metrics in metrics.items():
            averaged_metrics[method] = {}
            for metric_name, metric_values in method_metrics.items():
                tensor_values = torch.tensor(metric_values)
                is_valid = torch.isfinite(tensor_values) & ~torch.isnan(tensor_values)
                valid_values = tensor_values[is_valid]  # Exclude both 'inf' and 'nan' values
                print(f'{method} {metric_name}: {torch.tensor(valid_values).mean()}')
                averaged_value = float(valid_values.mean()) if valid_values.nelement() > 0 else float('nan')  # Avoid potential empty tensor
                averaged_metrics[method][metric_name] = averaged_value

        # Save to a JSON file
        filename_json = f'test_results/{self.dataset_name}/baseline/{self.split}/metrics.json'
        with open(filename_json, 'w') as json_file:
            json.dump(averaged_metrics, json_file, indent=4)

    def detection(self, num_eval=None):
        """
        Baseline DSP for detection using cosine similarity of delay signals or deconvolved signals
        """

        cs_deconv_list = []
        cs_delay_list = []
        cs_receiver_list = []
        positive_list = []

        loop_length = num_eval if num_eval is not None else len(self.room_id_pairs)
        for room_id_pair in tqdm(self.room_id_pairs[:loop_length]):
            room, id = room_id_pair
            data_dir = f'{self.dataset_dir}/{room}/{id}'

            metadata = torch.load(f'{data_dir}/metadata.pt')
            grid_points = metadata['grid_points']
            num_points = grid_points.shape[0]
            source1_idx = metadata['source1_idx']
            source2_idx = metadata['source2_idx']
            source_idx_list = [source1_idx, source2_idx]
            receiver_idx_list = metadata['receiver_idx_list']
            receiver_audio = self.load_audio_list(f'{data_dir}/receiver/receiver')

            for query_idx in range(num_points):
                if (query_idx not in receiver_idx_list):
                    deconv_audio = self.load_audio_list(f'{data_dir}/wiener/wiener{query_idx}')

                    # compute delay from distance
                    c = 343
                    sample_rate = 48000
                    query_point = grid_points[query_idx]
                    delay_list = []
                    for receiver_idx in receiver_idx_list:
                        receiver_point = grid_points[receiver_idx]
                        dist = (query_point - receiver_point).norm() - 0.1  # soundspaces offset
                        delay_list.append(int(dist / c * sample_rate))

                    delayed_audio = compute_delayed(receiver_audio, delay_list)

                    cs = cs_list(deconv_audio)
                    cs_delay = cs_list(delayed_audio)
                    cs_receiver = cs_list(receiver_audio)
                    if query_idx in source_idx_list:
                        positive_list.append(True)
                    else:
                        positive_list.append(False)

                    cs_deconv_list.append(cs)
                    cs_delay_list.append(cs_delay)
                    cs_receiver_list.append(cs_receiver)

        cs_deconv_list = torch.tensor(cs_deconv_list)
        cs_delay_list = torch.tensor(cs_delay_list)
        cs_receiver_list = torch.tensor(cs_receiver_list)
        positive_list = torch.tensor(positive_list)
        print(f'deconv_p: {cs_deconv_list[positive_list].mean()}')
        print(f'deconv_n: {cs_deconv_list[~positive_list].mean()}')
        print(f'delay_p: {cs_delay_list[positive_list].mean()}')
        print(f'delay_n: {cs_delay_list[~positive_list].mean()}')
        print(f'receiver_p: {cs_receiver_list[positive_list].mean()}')
        print(f'receiver_n: {cs_receiver_list[~positive_list].mean()}')

        # AUC
        metric = BinaryAUROC(thresholds=None)
        auc_deconv = metric(cs_deconv_list, positive_list)
        auc_delay = metric(cs_delay_list, positive_list)
        auc_receiver = metric(cs_receiver_list, positive_list)
        print(f'AUC (receiver, delay, deconv): {auc_receiver}, {auc_delay}, {auc_deconv}')

        # # search best threshold
        # cs_deconv_p_tensor = torch.tensor(cs_deconv_p_list)
        # cs_deconv_n_tensor = torch.tensor(cs_deconv_n_list)
        # th_list = torch.arange(-0.1, 0.3, 0.01)
        # accuracy_list = []
        # for th in th_list:
        #     accuracy_p = (cs_deconv_p_tensor > th).sum()
        #     accuracy_n = (cs_deconv_n_tensor <= th).sum()
        #     accuracy_ = accuracy_p + accuracy_n
        #     accuracy = float(accuracy_) / (len(cs_deconv_p_tensor) + len(cs_deconv_n_tensor))
        #     accuracy_list.append(accuracy)

        # max_accuracy = torch.tensor(accuracy_list).max()
        # max_idx = accuracy_list.index(max_accuracy)
        # max_th = th_list[max_idx]

        # print(f'max_accuracy: {max_accuracy}')
        # print(f'max_th: {max_th}')
        # # print(accuracy_list)

        # accuracy_p = (cs_deconv_p_tensor > max_th).float().mean()
        # accuracy_n = (cs_deconv_n_tensor <= max_th).float().mean()
        # print(accuracy_p)
        # print(accuracy_n)

        # max_th


if __name__ == "__main__":
    random.seed(42)

    # Config parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='separation')  # separation, detection
    parser.add_argument('--split', type=str, default='demo')
    parser.add_argument('--num_receivers', type=int, default=4)
    parser.add_argument('--audio_format', type=str, default='flac')
    parser.add_argument('--dataset_name', type=str, default='nvas3d_demo')
    parser.add_argument('--num_eval', type=int, default=None)  # None to test all data
    args = parser.parse_args()

    # Run baseline
    baseline = BaselineDSP(args.split, args.num_receivers, args.audio_format, args.dataset_name)
    if args.task == 'separation':
        baseline.separation(args.num_eval)
    elif args.task == 'detection':
        baseline.detection(args.num_eval)
