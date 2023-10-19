
import torchaudio
from torchaudio.transforms import Resample
import os


def process_directory(input_dir, output_dir, sample_rate=48000):
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".flac"):
                input_file_path = os.path.join(dirpath, filename)
                waveform_, sr = torchaudio.load(input_file_path)

                # Resample
                upsample_transform = Resample(sr, sample_rate)
                waveform = upsample_transform(waveform_)

                relative_path = os.path.relpath(dirpath, input_dir)
                clip_output_dir = os.path.join(output_dir, relative_path)

                os.makedirs(clip_output_dir, exist_ok=True)

                output_file_path = os.path.join(clip_output_dir, filename)
                torchaudio.save(output_file_path, waveform, sample_rate)


# Modify the paths and parameters as per your needs
input_dir = "data/source/LibriSpeech/"
output_dir = "data/source/LibriSpeech48k/"

process_directory(input_dir, output_dir)
