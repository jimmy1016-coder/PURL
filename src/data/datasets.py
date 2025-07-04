import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from data.data_utils import crop_or_pad, load_spk2utt, load_utt2spk, read_voxceleb_pairs_txt, to_one_hot


class VoxCelebContrastiveDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        wav_processor: nn.Module,
        feature_extractor: nn.Module,
        sample_rate: int,
        segment_length: int,
        samples_per_epoch: int,
        allow_segment_overlap: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.all_filenames = list(data_dir.glob("**/*.wav"))
        self.wav_processor = wav_processor
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.samples_per_epoch = samples_per_epoch
        self.allow_segment_overlap = allow_segment_overlap

    def __len__(self):
        return self.samples_per_epoch

    def _get_crops_with_overlap(self, audio):
        # get 2 random crops (no worry about overlap)
        seg1 = crop_or_pad(audio, self.segment_length)
        seg2 = crop_or_pad(audio, self.segment_length)

        return seg1, seg2

    def _get_crops_without_overlap(self, audio):
        audio_length = audio.shape[-1]
        max_offset = audio_length - (2 * self.segment_length)

        if max_offset > 0:
            # divide max offset between 2 segments
            seg1_offset = np.random.randint(low=0, high=max_offset)
            seg2_offset = np.random.randint(low=0, high=max_offset - seg1_offset)

            seg1_start_idx, seg1_end_idx = (
                seg1_offset,
                seg1_offset + self.segment_length,
            )
            seg2_start_idx, seg2_end_idx = (
                seg1_end_idx + seg2_offset,
                seg1_end_idx + seg2_offset + self.segment_length,
            )

            seg1 = audio[:, seg1_start_idx:seg1_end_idx]
            seg2 = audio[:, seg2_start_idx:seg2_end_idx]
        elif max_offset < 0:
            # if audio is shorter than 2*segment_length
            # we select a range in the middle to split
            # the audio and later pad the 2 resulting
            # segments to segment_length
            split_half_range = 0.1 * audio_length
            split_idx = (audio_length // 2) + np.random.randint(low=-1 * split_half_range, high=split_half_range)
            seg1 = audio[:, :split_idx]
            seg2 = audio[:, split_idx:]

            seg1 = crop_or_pad(seg1, self.segment_length)
            seg2 = crop_or_pad(seg2, self.segment_length)
        else:
            # this means audio_length is exactly 2*segment_length
            assert audio_length == 2 * self.segment_length, "Segment len bug"
            seg1 = audio[:, : self.segment_length]
            seg2 = audio[:, self.segment_length :]

        return seg1, seg2

    def _preprocess_file(self, signal):
        audio = self.wav_processor(signal)
        features = self.feature_extractor(audio)

        return features, audio

    def _get_crops(self, audio):
        if self.allow_segment_overlap:
            seg1, seg2 = self._get_crops_with_overlap(audio)
        else:
            seg1, seg2 = self._get_crops_without_overlap(audio)

        return seg1, seg2

    def __getitem__(self, index):
        filename = random.choice(self.all_filenames)
        audio, sr = torchaudio.load(filename, channels_first=True, backend="soundfile")

        if sr != self.sample_rate:
            audio = F.resample(audio, orig_freq=sr, new_freq=self.sample_rate)

        crop1, crop2 = self._get_crops(audio)

        features1, audio1 = self._preprocess_file(crop1)
        features2, audio2 = self._preprocess_file(crop2)

        stacked_features = torch.cat([features1, features2], dim=0)
        stacked_audios = torch.cat([audio1, audio2], dim=0)

        return stacked_features, stacked_audios


class VoxCelebSupConDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        spk2utt_file_path: str,
        valid_spk_ids: set,
        wav_processor: nn.Module,
        feature_extractor: nn.Module,
        sample_rate: int,
        segment_length: int,
        samples_per_epoch: int,
    ):
        self.data_dir = Path(data_dir)
        self.wav_processor = wav_processor
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.samples_per_epoch = samples_per_epoch

        self.spk2utt = load_spk2utt(spk2utt_file_path)

        # remove valid indices from spk2utt
        for id in valid_spk_ids:
            self.spk2utt.pop(id)

        self.speaker_ids = list(self.spk2utt.keys())
        self.label_encoder = {label: k for k, label in enumerate(self.speaker_ids)}

    def __len__(self):
        return self.samples_per_epoch

    def _get_files(self):
        speaker_id = random.choice(self.speaker_ids)
        speaker_files = self.spk2utt[speaker_id]
        file1, file2 = np.random.choice(speaker_files, size=2, replace=False)

        return file1, file2, speaker_id

    def _load_and_preprocess_file(self, filename):
        audio, sr = torchaudio.load(uri=self.data_dir / filename, channels_first=True, backend="soundfile")

        if sr != self.sample_rate:
            audio = F.resample(audio, orig_freq=sr, new_freq=self.sample_rate)

        audio = crop_or_pad(audio, self.segment_length)

        audio = self.wav_processor(audio)
        spec = self.feature_extractor(audio)

        return spec, audio

    def __getitem__(self, index):
        file1, file2, spk_id = self._get_files()

        features1, audio1 = self._load_and_preprocess_file(file1)
        features2, audio2 = self._load_and_preprocess_file(file2)

        stacked_features = torch.cat([features1, features2], dim=0)
        stacked_audios = torch.cat([audio1, audio2], dim=0)

        label = self.label_encoder[spk_id]

        return stacked_features, stacked_audios, label


class VoxCelebSupConDatasetForBatchSampler(VoxCelebSupConDataset):
    def _get_files_by_id(self, speaker_id):
        speaker_files = self.spk2utt[speaker_id]
        file1, file2 = np.random.choice(speaker_files, size=2, replace=False)

        return file1, file2

    def __getitem__(self, speaker_id):
        file1, file2 = self._get_files_by_id(speaker_id)

        features1, audio1 = self._load_and_preprocess_file(file1)
        features2, audio2 = self._load_and_preprocess_file(file2)

        stacked_features = torch.cat([features1, features2], dim=0)
        stacked_audios = torch.cat([audio1, audio2], dim=0)

        label = self.label_encoder[speaker_id]

        return stacked_features, stacked_audios, label


class VoxCelebSupervisedDataset(Dataset):
    def __init__(
        self,
        data_dir,
        utt2spk_file_path,
        valid_spk_ids,
        wav_processor,
        feature_extractor,
        sample_rate,
        segment_length,
        samples_per_epoch,
    ):
        self.data_dir = Path(data_dir)
        self.utt2spk = load_utt2spk(utt2spk_file_path)
        self.wav_processor = wav_processor
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.samples_per_epoch = samples_per_epoch

        # remove files belonging to validation speakers
        self.files = [f for f, spk_id in self.utt2spk.items() if spk_id not in valid_spk_ids]

        self.speaker_ids = {spk_id for f, spk_id in self.utt2spk.items() if spk_id not in valid_spk_ids}
        self.label_encoder = {label: k for k, label in enumerate(self.speaker_ids)}

    def __len__(self):
        return self.samples_per_epoch

    def _load_and_preprocess_file(self, filename):
        audio, sr = torchaudio.load(uri=self.data_dir / filename, channels_first=True, backend="soundfile")

        if sr != self.sample_rate:
            audio = F.resample(audio, orig_freq=sr, new_freq=self.sample_rate)

        audio = crop_or_pad(audio, self.segment_length)

        audio = self.wav_processor(audio)
        spec = self.feature_extractor(audio)

        return spec, audio

    def __getitem__(self, index):
        idx = random.randint(0, len(self.files) - 1)
        filename = self.files[idx]
        spk_id = self.utt2spk[filename]

        label = self.label_encoder[spk_id]
        label = to_one_hot(label, len(self.label_encoder.items()))

        features, audio = self._load_and_preprocess_file(filename)

        audio = self.wav_processor(audio)
        features = self.feature_extractor(audio)

        return features.squeeze(), audio, label


class VoxCelebEvalDataset(Dataset):
    def __init__(self, data_dir, pairs_txt_file, processor, sample_rate, segment_length=None):
        self.data_dir = Path(data_dir)
        self.pairs_txt_file = pairs_txt_file
        self.labels, self.files1, self.files2 = read_voxceleb_pairs_txt(self.pairs_txt_file)
        self.paths = np.unique(np.concatenate((self.files1, self.files2)))
        self.processor = processor
        self.sample_rate = sample_rate
        self.segment_length = segment_length

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        filename = self.paths[index]
        audio, sr = torchaudio.load(uri=self.data_dir / filename, channels_first=True, backend="soundfile")

        if sr != self.sample_rate:
            audio = F.resample(audio, orig_freq=sr, new_freq=self.sample_rate)

        if self.segment_length:
            audio = crop_or_pad(audio, self.segment_length)

        if self.processor:
            signal = self.processor(audio)

        return signal, filename


class VoxCelebSingleSpeakerDataset(Dataset):
    def __init__(
        self,
        data_dir,
        spk2utt_file_path,
        wav_processor,
        feature_extractor,
        sample_rate,
        segment_length,
        files_per_speaker,
        exclude_speaker_ids,
    ):
        self.data_dir = data_dir
        self.spk2utt = load_spk2utt(spk2utt_file_path)
        self.wav_processor = wav_processor
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.files_per_speaker = files_per_speaker

        self.speaker_ids = [Path(d).name for d in self.spk2utt.keys() if Path(d).name not in exclude_speaker_ids]

    def __len__(self):
        return len(self.speaker_ids)

    def _get_files(self, speaker_id):
        speaker_files = self.spk2utt[speaker_id]
        if self.files_per_speaker > len(speaker_files):
            print(f"Speaker {speaker_id}, doesn't have enough samples, using {len(speaker_files)} files.")
            num_files_to_choose = len(speaker_files)
        else:
            num_files_to_choose = self.files_per_speaker

        files = np.random.choice(speaker_files, size=num_files_to_choose, replace=False)

        return files

    def _load_and_preprocess_file(self, filename):
        audio, sr = torchaudio.load(uri=self.data_dir / filename, channels_first=True, backend="soundfile")

        if sr != self.sample_rate:
            audio = F.resample(audio, orig_freq=sr, new_freq=self.sample_rate)

        if self.segment_length:
            audio = crop_or_pad(audio, self.segment_length)

        audio = self.wav_processor(audio)
        spec = self.feature_extractor(audio)

        return spec, audio

    def __getitem__(self, index):
        speaker_id = self.speaker_ids[index]
        files = self._get_files(speaker_id)

        specs = []

        for file in files:
            spec, _ = self._load_and_preprocess_file(file)
            specs.append(spec.squeeze())

        stacked_specs = torch.stack(specs, dim=0)

        return stacked_specs, speaker_id
