from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch.nn as nn
import numpy as np
from typing import Optional
import math

from data import (
    VoxCelebContrastiveDataset,
    VoxCelebSupConDataset,
    VoxCelebSupConDatasetForBatchSampler,
    VoxCelebSupervisedDataset,
    VoxCelebEvalDataset,
    VoxCelebSingleSpeakerDataset
)
from data.feature_extractors import FeatureExtractor
from data.data_utils import read_voxceleb_pairs_txt
from data.samplers import ClusterBatchSampler, UniqueBatchSampler


@dataclass
class VoxCelebContrastiveSplitConfig:
    data_dir: Path
    wav_processor: nn.Module
    feature_extractor: FeatureExtractor
    sample_rate: int
    segment_length: Optional[int]
    samples_per_epoch: int
    allow_segment_overlap: bool
    batch_size: int
    num_workers: int


@dataclass
class VoxCelebSupConTrainDataConfig:
    data_dir: Path
    spk2utt_file_path: Path
    wav_processor: nn.Module
    feature_extractor: FeatureExtractor
    sample_rate: int
    segment_length: Optional[int]
    samples_per_epoch: int
    batch_size: int
    num_workers: int


@dataclass
class VoxCelebSupervisedTrainDataConfig:
    data_dir: Path
    utt2spk_file_path: Path
    wav_processor: nn.Module
    feature_extractor: FeatureExtractor
    sample_rate: int
    segment_length: Optional[int]
    samples_per_epoch: int
    batch_size: int
    num_workers: int


@dataclass
class ClusterBatchSamplerConfig:
    cluster_dict_path: str
    hard_ratio: float
    allow_repeat_speakers: bool = False
    allow_multi_cluster: bool = False
    cluster_sims_path: Optional[str] = None
    sample_unique_rest: bool = False


@dataclass
class VoxCelebEvalDataConfig:
    data_dir: Path
    trials_file_path: Path
    feature_extractor: FeatureExtractor
    sample_rate: int
    segment_length: Optional[int]
    batch_size: int
    num_workers: int


@dataclass
class VoxCelebClusteringDataConfig:
    data_dir: Path
    spk2utt_file_path: Path
    wav_processor: nn.Module
    feature_extractor: FeatureExtractor
    sample_rate: int
    segment_length: Optional[int]
    files_per_speaker: int
    batch_size: int
    num_workers: int


class VoxCelebContrastiveDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_config: VoxCelebContrastiveSplitConfig,
        valid_config: VoxCelebContrastiveSplitConfig,
        test_config: VoxCelebEvalDataConfig,
    ):
        super().__init__()
        self.train_config = train_config
        self.valid_config = valid_config
        self.test_config = test_config

    def _create_dataloader(self, config: VoxCelebContrastiveSplitConfig):
        dataset = VoxCelebContrastiveDataset(
            data_dir=config.data_dir,
            wav_processor=config.wav_processor,
            feature_extractor=config.feature_extractor,
            sample_rate=config.sample_rate,
            segment_length=config.segment_length,
            samples_per_epoch=config.samples_per_epoch // 2,
            allow_segment_overlap=config.allow_segment_overlap,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size // 2,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        return dataloader

    def train_dataloader(self):
        return self._create_dataloader(self.train_config)

    def val_dataloader(self):
        return self._create_dataloader(self.valid_config)

    def test_dataloader(self):
        dataset = VoxCelebEvalDataset(
            data_dir=self.test_config.data_dir,
            pairs_txt_file=self.test_config.trials_file_path,
            processor=self.test_config.feature_extractor,
            segment_length=self.test_config.segment_length,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.test_config.batch_size,
            num_workers=self.test_config.num_workers,
            pin_memory=True,
        )

        return dataloader


class VoxCelebSupConDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_config: VoxCelebSupConTrainDataConfig,
        valid_config: VoxCelebEvalDataConfig,
        test_config: VoxCelebEvalDataConfig,
        clustering_config: Optional[VoxCelebClusteringDataConfig] = None,
        batch_sampler_config: Optional[ClusterBatchSamplerConfig] = None,
        use_unique_batch_sampler: bool = False,
    ):
        super().__init__()
        self.train_config = train_config
        self.valid_config = valid_config
        self.test_config = test_config
        self.clustering_config = clustering_config
        self.batch_sampler_config = batch_sampler_config
        self.use_unique_batch_sampler = use_unique_batch_sampler

        _, files1, files2 = read_voxceleb_pairs_txt(valid_config.trials_file_path)
        valid_file_paths = np.unique(np.concatenate((files1, files2)))
        self.valid_spk_ids = {p.split("/")[-3] for p in valid_file_paths}

    def _create_train_dataloader_with_batch_sampler(self):
        dataset = VoxCelebSupConDatasetForBatchSampler(
            data_dir=self.train_config.data_dir,
            spk2utt_file_path=self.train_config.spk2utt_file_path,
            valid_spk_ids=self.valid_spk_ids,
            wav_processor=self.train_config.wav_processor,
            feature_extractor=self.train_config.feature_extractor,
            sample_rate=self.train_config.sample_rate,
            segment_length=self.train_config.segment_length,
            samples_per_epoch=self.train_config.samples_per_epoch,
        )

        if self.use_unique_batch_sampler:
            batch_sampler = UniqueBatchSampler(
                spk2utt_file_path=self.train_config.spk2utt_file_path,
                valid_spk_ids=self.valid_spk_ids,
                batch_size=self.train_config.batch_size // 2,
                steps_per_epoch=math.ceil(self.train_config.samples_per_epoch / self.train_config.batch_size),
            )
        else:
            batch_sampler = ClusterBatchSampler(
                cluster_dict_path=self.batch_sampler_config.cluster_dict_path,
                batch_size=self.train_config.batch_size // 2,
                steps_per_epoch=math.ceil(self.train_config.samples_per_epoch / self.train_config.batch_size),
                hard_ratio=self.batch_sampler_config.hard_ratio,
                allow_repeat_speakers=self.batch_sampler_config.allow_repeat_speakers,
                allow_multi_cluster=self.batch_sampler_config.allow_multi_cluster,
                cluster_sims_path=self.batch_sampler_config.cluster_sims_path,
                sample_unique_rest=self.batch_sampler_config.sample_unique_rest,
            )

        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=self.train_config.num_workers,
            pin_memory=True,
        )

        return dataloader

    def _create_train_dataloader_without_batch_sampler(self):
        dataset = VoxCelebSupConDataset(
            data_dir=self.train_config.data_dir,
            spk2utt_file_path=self.train_config.spk2utt_file_path,
            valid_spk_ids=self.valid_spk_ids,
            wav_processor=self.train_config.wav_processor,
            feature_extractor=self.train_config.feature_extractor,
            sample_rate=self.train_config.sample_rate,
            segment_length=self.train_config.segment_length,
            samples_per_epoch=self.train_config.samples_per_epoch // 2,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.train_config.batch_size // 2,
            num_workers=self.train_config.num_workers,
            pin_memory=True,
        )

        return dataloader

    def _create_eval_dataloader(self, config: VoxCelebEvalDataConfig):
        dataset = VoxCelebEvalDataset(
            data_dir=config.data_dir,
            pairs_txt_file=config.trials_file_path,
            processor=config.feature_extractor,
            sample_rate=config.sample_rate,
            segment_length=config.segment_length,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        return dataloader

    def train_dataloader(self):
        if self.batch_sampler_config or self.use_unique_batch_sampler:
            return self._create_train_dataloader_with_batch_sampler()
        else:
            return self._create_train_dataloader_without_batch_sampler()

    def val_dataloader(self):
        return self._create_eval_dataloader(self.valid_config)

    def test_dataloader(self):
        return self._create_eval_dataloader(self.test_config)

    def predict_dataloader(self):
        if self.clustering_config is None:
            return None

        dataset = VoxCelebSingleSpeakerDataset(
            data_dir=self.clustering_config.data_dir,
            spk2utt_file_path=self.clustering_config.spk2utt_file_path,
            wav_processor=self.clustering_config.wav_processor,
            feature_extractor=self.clustering_config.feature_extractor,
            sample_rate=self.clustering_config.sample_rate,
            segment_length=self.clustering_config.segment_length,
            files_per_speaker=self.clustering_config.files_per_speaker,
            exclude_speaker_ids=self.valid_spk_ids
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.clustering_config.batch_size,
            num_workers=self.clustering_config.num_workers,
            pin_memory=True,
        )

        return dataloader


class VoxCelebSupervisedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_config: VoxCelebSupervisedTrainDataConfig,
        valid_config: VoxCelebEvalDataConfig,
        test_config: VoxCelebEvalDataConfig,
        clustering_config: VoxCelebClusteringDataConfig,
    ):
        super().__init__()
        self.train_config = train_config
        self.valid_config = valid_config
        self.test_config = test_config
        self.clustering_config = clustering_config

        _, files1, files2 = read_voxceleb_pairs_txt(valid_config.trials_file_path)
        valid_file_paths = np.unique(np.concatenate((files1, files2)))
        self.valid_spk_ids = {p.split("/")[-3] for p in valid_file_paths}

    def train_dataloader(self):
        dataset = VoxCelebSupervisedDataset(
            data_dir=self.train_config.data_dir,
            utt2spk_file_path=self.train_config.utt2spk_file_path,
            valid_spk_ids=self.valid_spk_ids,
            wav_processor=self.train_config.wav_processor,
            feature_extractor=self.train_config.feature_extractor,
            sample_rate=self.train_config.sample_rate,
            segment_length=self.train_config.segment_length,
            samples_per_epoch=self.train_config.samples_per_epoch,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
            pin_memory=True,
        )

        return dataloader

    def _create_eval_dataloader(self, config: VoxCelebEvalDataConfig):
        dataset = VoxCelebEvalDataset(
            data_dir=config.data_dir,
            pairs_txt_file=config.trials_file_path,
            processor=config.feature_extractor,
            sample_rate=config.sample_rate,
            segment_length=config.segment_length,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        return dataloader

    def val_dataloader(self):
        return self._create_eval_dataloader(self.valid_config)

    def test_dataloader(self):
        return self._create_eval_dataloader(self.test_config)

    def predict_dataloader(self):
        dataset = VoxCelebSingleSpeakerDataset(
            data_dir=self.clustering_config.data_dir,
            spk2utt_file_path=self.clustering_config.spk2utt_file_path,
            wav_processor=self.clustering_config.wav_processor,
            feature_extractor=self.clustering_config.feature_extractor,
            sample_rate=self.clustering_config.sample_rate,
            segment_length=self.clustering_config.segment_length,
            files_per_speaker=self.clustering_config.files_per_speaker,
            exclude_speaker_ids=self.valid_spk_ids
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.clustering_config.batch_size,
            num_workers=self.clustering_config.num_workers,
            pin_memory=True,
        )

        return dataloader
