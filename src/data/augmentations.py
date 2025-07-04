from pathlib import Path
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from speechbrain.processing.signal_processing import reverberate

from data.data_utils import crop_or_pad


class Compose(nn.Module):
    def __init__(self, transforms: list[nn.Module], shuffle: bool = False):
        super().__init__()
        self.transforms = transforms
        self.shuffle = shuffle

    def forward(self, data):
        if self.shuffle:
            random.shuffle(self.transforms)

        for transform in self.transforms:
            data = transform(data)

        return data


class OneOf(nn.Module):
    def __init__(self, transforms: list[nn.Module]):
        super().__init__()
        self.transforms = transforms

    def forward(self, data):
        transform = random.choice(self.transforms)

        return transform(data)


class WavAddNoise(nn.Module):
    def __init__(
        self,
        noise_dirs: list[str],
        file_extension: str,
        snr_low_db: int,
        snr_high_db: int,
        sample_rate: int,
    ):
        super().__init__()
        self.snr_low_db = snr_low_db
        self.snr_high_db = snr_high_db
        self.files = []
        for nd in noise_dirs:
            self.files.extend(list(Path(nd).glob(f"**/*.{file_extension}")))
        self.sample_rate = sample_rate

    def scale_to_snr(self, audio, noise):
        snr = np.random.randint(self.snr_low_db, self.snr_high_db)

        clean_db = 10 * torch.log10(torch.mean(audio.squeeze() ** 2) + 1e-5)
        noise_db = 10 * torch.log10(torch.mean(noise.squeeze() ** 2) + 1e-5)
        noise_scale = torch.sqrt(10 ** ((clean_db - noise_db - snr) / 10))

        return noise * noise_scale + audio

    def forward(self, audio):
        clean_audio_max = torch.max(torch.abs(audio))

        file = random.choice(self.files)
        noise, sr = torchaudio.load(file, channels_first=True, backend="soundfile")
        # make sure audio is mono
        noise = noise.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            noise = torchaudio.functional.resample(noise, sr, self.sample_rate)

        noise = crop_or_pad(noise, audio.shape[-1])

        audio = self.scale_to_snr(audio, noise)

        # make sure the amplitude of the audio stays the same
        audio /= torch.max(torch.abs(audio))
        audio *= clean_audio_max

        return audio


class WavAddRIR(nn.Module):
    def __init__(self, rir_pkl_file_path: str):
        super().__init__()
        with open(rir_pkl_file_path, "rb") as f:
            rirs = pickle.load(f)

        self.rirs = [torch.Tensor(rir).unsqueeze(0) for rir in rirs]

    def forward(self, audio):
        original_audio_len = audio.shape[-1]

        rir = random.choice(self.rirs)

        audio = reverberate(audio, rir)

        # make sure the amplitude of the audio stays the same
        # audio /= torch.max(torch.abs(audio))
        # audio += original_audio_max

        return audio[:, :original_audio_len]
        # return audio


class SpecNormalization(nn.Module):
    def forward(self, spec):
        spec = spec - torch.mean(spec, dim=-2)

        return spec
